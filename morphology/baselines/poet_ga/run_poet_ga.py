import os
import numpy as np
import shutil
import random
import math
import copy
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..', '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo.run_ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import (
    get_percent_survival_evals, mutate, TerminationCondition, Structure
)


def _do_transfer(task_states, tasks, transfer_num, args, num_cores,
                  home_path, generation, prescreening_iters):
    """
    Cross-task morphology transfer with pre-screening (POET-inspired).

    Unlike POET where controllers can be directly evaluated across environments
    (same body, different terrain), morphology transfer requires retraining a
    controller from scratch because different bodies have different obs/act spaces.

    To avoid wasting full training budget on incompatible morphologies, we run a
    SHORT PPO pre-screening on the target task first. Only candidates that show
    promise (fitness > worst current individual) are accepted into the population.
    """
    print(f'\n--- Cross-task Morphology Transfer (pre-screening: {prescreening_iters} iters) ---')

    task_elites = {}
    for task in tasks:
        ts = task_states[task]
        trained = [s for s in ts['structures'] if s.is_survivor]
        task_elites[task] = trained[:transfer_num]

    candidates = []
    for target_task in tasks:
        ts = task_states[target_task]
        if len(ts['structures']) < 3:
            continue

        for source_task in tasks:
            if source_task == target_task:
                continue

            for elite in task_elites[source_task]:
                h = hashable(elite.body)
                if h in ts['hashes']:
                    continue

                candidates.append({
                    'source_task': source_task,
                    'target_task': target_task,
                    'body': elite.body.copy(),
                    'connections': elite.connections.copy(),
                    'source_fitness': elite.fitness,
                    'hash': h,
                })

    if not candidates:
        print('  No transfer candidates (all duplicates). Skipping.')
        return []

    print(f'  Pre-screening {len(candidates)} candidates ...')

    # ---- Pre-screen all candidates with short PPO ----
    tc_screen = TerminationCondition(prescreening_iters)
    screen_path = os.path.join(home_path, f"_transfer_screen_gen_{generation}")
    os.makedirs(screen_path, exist_ok=True)

    screen_structures = []
    for i, cand in enumerate(candidates):
        ts = task_states[cand['target_task']]
        s = Structure(cand['body'], cand['connections'], i, ts['task_idx'])
        screen_structures.append(s)

    group = mp.Group()
    for i, (cand, structure) in enumerate(zip(candidates, screen_structures)):
        ppo_args = (
            i % 2, args, cand['target_task'], structure,
            tc_screen, (screen_path, i)
        )
        group.add_job(run_ppo, ppo_args, callback=structure.set_reward)

    group.run_jobs(num_cores)

    for s in screen_structures:
        s.compute_fitness()

    # ---- Decide which transfers to accept ----
    transfer_log = []

    for cand, screen_s in zip(candidates, screen_structures):
        target_ts = task_states[cand['target_task']]
        structures = target_ts['structures']

        worst_idx = -1
        for idx in range(len(structures) - 1, -1, -1):
            if not structures[idx].is_survivor:
                worst_idx = idx
                break

        if worst_idx == -1:
            continue

        worst_fitness = structures[worst_idx].fitness

        if screen_s.fitness > worst_fitness:
            h = cand['hash']
            if h in target_ts['hashes']:
                continue

            structures[worst_idx] = Structure(
                cand['body'], cand['connections'],
                worst_idx, target_ts['task_idx'],
            )
            structures[worst_idx].is_survivor = False
            structures[worst_idx].prev_gen_label = -1

            target_ts['hashes'][h] = True

            msg = (f'  ACCEPTED: {cand["source_task"]} -> {cand["target_task"]} '
                   f'(src_fit={cand["source_fitness"]:.2f}, '
                   f'screen_fit={screen_s.fitness:.2f}, '
                   f'replaced={worst_fitness:.2f})')
            transfer_log.append(msg)
        else:
            print(f'  REJECTED: {cand["source_task"]} -> {cand["target_task"]} '
                  f'(src_fit={cand["source_fitness"]:.2f}, '
                  f'screen_fit={screen_s.fitness:.2f}, '
                  f'worst={worst_fitness:.2f})')

    shutil.rmtree(screen_path, ignore_errors=True)

    for msg in transfer_log:
        print(msg)
    print(f'  Accepted: {len(transfer_log)} / {len(candidates)} candidates')
    return transfer_log


def run_poet_ga(
    args,
    tasks,
    structure_shape,
    pop_size,
    max_evaluations,
    train_iters,
    num_cores,
    experiment_name,
    transfer_interval=5,
    transfer_num=2,
    prescreening_iters=None,
):
    if prescreening_iters is None:
        prescreening_iters = max(50, train_iters // 5)

    print(f'\n{"="*60}')
    print(f'POET+GA: Multi-task GA with Cross-task Transfer')
    print(f'Tasks: {tasks}')
    print(f'Pop size: {pop_size}, Max evals per task: {max_evaluations}')
    print(f'Train iters: {train_iters}, Num cores: {num_cores}')
    print(f'Transfer every {transfer_interval} gens, top {transfer_num}')
    print(f'Pre-screening iters: {prescreening_iters}')
    print(f'{"="*60}\n')

    home_path = os.path.join(root_dir, "saved_data", experiment_name)

    try:
        os.makedirs(home_path)
    except FileExistsError:
        print(f'Experiment ({experiment_name}) already exists.')
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            os.makedirs(home_path)
        else:
            return

    with open(os.path.join(home_path, "metadata.txt"), "w") as f:
        f.write(f'TASKS: {" ".join(tasks)}\n')
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.write(f'TRANSFER_INTERVAL: {transfer_interval}\n')
        f.write(f'TRANSFER_NUM: {transfer_num}\n')
        f.write(f'PRESCREENING_ITERS: {prescreening_iters}\n')

    tc = TerminationCondition(train_iters)

    # ---- Initialize per-task populations ----
    task_states = {}
    for task_idx, task in enumerate(tasks):
        structures = []
        hashes = {}
        for i in range(pop_size):
            temp = sample_robot(structure_shape)
            while hashable(temp[0]) in hashes:
                temp = sample_robot(structure_shape)
            structures.append(Structure(*temp, i, task_idx))
            hashes[hashable(temp[0])] = True

        task_states[task] = {
            'structures': structures,
            'hashes': hashes,
            'num_evaluations': pop_size,
            'task_idx': task_idx,
        }

    generation = 0
    all_transfer_logs = []

    while True:
        if all(ts['num_evaluations'] >= max_evaluations
               for ts in task_states.values()):
            break

        print(f'\n{"="*60}')
        print(f'GENERATION {generation}')
        print(f'{"="*60}')

        # ==========================================================
        # PHASE 1: Train controllers & evaluate for every active task
        # ==========================================================
        for task in tasks:
            ts = task_states[task]
            if ts['num_evaluations'] >= max_evaluations:
                print(f'[{task}] Already finished ({ts["num_evaluations"]} evals). Skipping.')
                continue

            structures = ts['structures']
            task_idx = ts['task_idx']

            gen_path = os.path.join(home_path, task, f"generation_{generation}")
            save_path_structure = os.path.join(gen_path, "structure")
            save_path_controller = os.path.join(gen_path, "controller")
            os.makedirs(save_path_structure, exist_ok=True)
            os.makedirs(save_path_controller, exist_ok=True)

            for s in structures:
                np.savez(
                    os.path.join(save_path_structure, str(s.label)),
                    s.body, s.connections
                )

            group = mp.Group()
            for structure in structures:
                if structure.is_survivor:
                    src = os.path.join(
                        home_path, task,
                        f"generation_{generation - 1}", "controller",
                        f"task-{task_idx}_robot-{structure.prev_gen_label}_controller.pt"
                    )
                    dst = os.path.join(
                        save_path_controller,
                        f"task-{task_idx}_robot-{structure.label}_controller.pt"
                    )
                    print(f'[{task}] Copying controller for survivor {structure.label}')
                    try:
                        shutil.copy(src, dst)
                    except Exception as e:
                        print(f'[{task}] Error copying controller: {e}')
                else:
                    ppo_args = (
                        structure.label % 2, args, task, structure,
                        tc, (save_path_controller, structure.label)
                    )
                    group.add_job(run_ppo, ppo_args, callback=structure.set_reward)

            group.run_jobs(num_cores)

            for s in structures:
                s.compute_fitness()
            structures.sort(key=lambda s: s.fitness, reverse=True)

            with open(os.path.join(gen_path, "output.txt"), "w") as f:
                for s in structures:
                    f.write(f"{s.label}\t\t{s.fitness}\t\t{s.prev_gen_label}\n")

            ts['structures'] = structures
            print(f'[{task}] Gen {generation} | Best: {structures[0].fitness:.2f} '
                  f'| Evals: {ts["num_evaluations"]}')

        # Early exit check
        if all(ts['num_evaluations'] >= max_evaluations
               for ts in task_states.values()):
            break

        # ==========================================================
        # PHASE 2: Selection & Mutation for each task
        # ==========================================================
        for task in tasks:
            ts = task_states[task]
            if ts['num_evaluations'] >= max_evaluations:
                continue

            structures = ts['structures']
            percent_survival = get_percent_survival_evals(
                ts['num_evaluations'], max_evaluations
            )
            num_survivors = max(2, math.ceil(pop_size * percent_survival))

            survivors = structures[:num_survivors]
            for i, s in enumerate(survivors):
                s.is_survivor = True
                s.prev_gen_label = s.label
                s.label = i

            num_children = 0
            while (num_children < (pop_size - num_survivors)
                   and ts['num_evaluations'] < max_evaluations):
                parent_idx = random.randint(0, num_survivors - 1)
                child, _ = mutate(
                    survivors[parent_idx].body.copy(),
                    mutation_rate=0.1, num_attempts=50
                )

                if child is not None and hashable(child[0]) not in ts['hashes']:
                    idx = num_survivors + num_children
                    structures[idx] = Structure(
                        *child, idx, ts['task_idx']
                    )
                    structures[idx].prev_gen_label = survivors[parent_idx].prev_gen_label
                    ts['hashes'][hashable(child[0])] = True
                    num_children += 1
                    ts['num_evaluations'] += 1

            ts['structures'] = structures[:num_survivors + num_children]

        # ==========================================================
        # PHASE 3: Cross-task morphology transfer
        # ==========================================================
        if generation > 0 and generation % transfer_interval == 0:
            logs = _do_transfer(
                task_states, tasks, transfer_num, args, num_cores,
                home_path, generation, prescreening_iters
            )
            all_transfer_logs.append({
                'generation': generation,
                'transfers': logs,
            })

            with open(os.path.join(home_path, "transfer_log.txt"), "a") as f:
                f.write(f"\n=== Generation {generation} ===\n")
                for msg in logs:
                    f.write(msg + "\n")

        # Progress log
        with open(os.path.join(home_path, "progress_log.txt"), "a") as f:
            for task in tasks:
                ts = task_states[task]
                best = ts['structures'][0].fitness if ts['structures'] else 0
                f.write(f"gen={generation}\ttask={task}\t"
                        f"best={best:.2f}\tevals={ts['num_evaluations']}\n")

        generation += 1

    # ---- Final summary ----
    print(f'\n{"="*60}')
    print(f'POET+GA COMPLETE  (Total generations: {generation})')
    print(f'{"="*60}')
    for task in tasks:
        ts = task_states[task]
        if ts['structures']:
            best = ts['structures'][0]
            print(f'[{task}] Best fitness: {best.fitness:.2f} | '
                  f'Total evals: {ts["num_evaluations"]}')
    print(f'Total cross-task transfers: '
          f'{sum(len(e["transfers"]) for e in all_transfer_logs)}')
    print(f'Results saved to: {home_path}')

    return task_states


def evaluate_on_target(
    args,
    task_states,
    training_tasks,
    target_task,
    pop_size,
    train_iters,
    num_cores,
    experiment_name,
):
    """
    Evaluate the final populations from POET training on an unseen target task.
    Collects the top `pop_size` morphologies from all training tasks' final
    populations (split evenly) and trains PPO from scratch on the target task.
    """
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    eval_dir = os.path.join(home_path, "target_eval")
    save_path_controller = os.path.join(eval_dir, "controller")
    os.makedirs(save_path_controller, exist_ok=True)

    n_tasks = len(training_tasks)
    per_task_counts = [pop_size // n_tasks] * n_tasks
    for i in range(pop_size % n_tasks):
        per_task_counts[i] += 1

    selected = []
    label = 0
    for task, count in zip(training_tasks, per_task_counts):
        ts = task_states[task]
        for s in ts['structures'][:count]:
            selected.append(Structure(s.body.copy(), s.connections.copy(), label, task_id=0))
            label += 1

    print(f'\n{"="*60}')
    print(f'Evaluating top {len(selected)} morphologies on {target_task}')
    print(f'(from training tasks: {", ".join(f"{t}:{c}" for t, c in zip(training_tasks, per_task_counts))})')
    print(f'{"="*60}\n')

    tc = TerminationCondition(train_iters)
    group = mp.Group()
    for structure in selected:
        ppo_args = (
            structure.label % 2, args, target_task, structure,
            tc, (save_path_controller, structure.label)
        )
        group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
    group.run_jobs(num_cores)

    for s in selected:
        s.compute_fitness()
    selected.sort(key=lambda s: s.fitness, reverse=True)

    with open(os.path.join(eval_dir, "output.txt"), "w") as f:
        f.write(f"# Target task: {target_task}, n={len(selected)}\n")
        for s in selected:
            f.write(f"{s.label}\t\t{s.fitness}\n")

    fits = [s.fitness for s in selected]
    print(f'\nTarget task {target_task}: mean={np.mean(fits):.4f}  '
          f'max={np.max(fits):.4f}  median={np.median(fits):.4f}')
    print(f'Results: {eval_dir}')
