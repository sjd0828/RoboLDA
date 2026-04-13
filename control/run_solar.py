"""SOLAR baseline: two-level Transformer with online AP clustering.

Uses model_lowrank.ATTBase (intra-organ then inter-organ attention) and
periodically updates organ assignments via Affinity Propagation based on
per-actuator ablation evaluation.

NOTE: model_lowrank.ATTBase takes (obs_dim, h_dim, ...) argument order,
which differs from metamorph_organ.ATTBase's (h_dim, obs_dim, ...).
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
from sklearn.cluster import AffinityPropagation

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)
sys.path.insert(1, os.path.join(curr_dir, "externals", "pytorch_a2c_ppo_acktr_gail"))

from ppo.arguments import get_args
from ppo.envs import make_vec_envs
from ppo import utils
from ppo.evaluate_universal import evaluate
from ppo.evaluate_universal_organ import evaluate as evaluate_ablation
from evogym.utils import get_full_connectivity

import utils.mp_group as mp
from utils.algo_utils import TerminationCondition, Structure

from a2c_ppo_acktr.model_lowrank import ATTBase, Policy
from a2c_ppo_acktr.algo.ppo_universal import PPO
from a2c_ppo_acktr.storage import RolloutStorage

import evogym.envs

HEAD_SIZES = {
    "Walker-v0": 2, "Carrier-v0": 6, "Pusher-v0": 6,
    "Climber-v0": 2, "UpStepper-v0": 14, "DownStepper-v0": 14,
    "BridgeWalker-v0": 3, "Catcher-v0": 7, "Balancer-v0": 1, "Thrower-v0": 6,
}


def parse_control_args():
    parser = argparse.ArgumentParser(
        description="SOLAR baseline: two-level Transformer with AP clustering")
    parser.add_argument("--task", type=str, default="Walker-v0",
                        help="EvoGym task name")
    parser.add_argument("--morph-dir", type=str, required=True,
                        help="Directory containing morph.pt")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--num-robots", type=int, default=10,
                        help="Number of robots to sample")
    parser.add_argument("--train-iters", type=int, default=2000,
                        help="PPO training iterations per robot")
    parser.add_argument("--num-cores", type=int, default=10,
                        help="Number of parallel worker processes")
    parser.add_argument("--h-dim", type=int, default=None,
                        help="Task observation head size (default: from HEAD_SIZES)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for Transformer")
    parser.add_argument("--obs-dim", type=int, default=8,
                        help="Per-voxel observation dimension")
    parser.add_argument("--action-dim", type=int, default=1,
                        help="Per-voxel action dimension")
    parser.add_argument("--n-head", type=int, default=1,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in Transformer")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Number of parallel environments per robot")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Rollout length per PPO update")
    parser.add_argument("--width", type=int, default=5,
                        help="Robot body grid width")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Evaluate every N PPO updates")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Log every N PPO updates")
    parser.add_argument("--ap-update-interval", type=int, default=100,
                        help="Update AP organ assignments every N PPO iters")
    parser.add_argument("--ap-affinity-scale", type=float, default=5.0,
                        help="Scale factor for position-based affinity")
    parser.add_argument("--ap-affinity-decay", type=float, default=0.1,
                        help="Decay rate for distance in affinity matrix")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Experiment output directory name")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA")
    return parser.parse_args()


def build_position_affinity(grid_size=5, decay=0.1, scale=5.0):
    """Build a position-based affinity matrix for a grid_size x grid_size body."""
    n = grid_size * grid_size
    affinity = np.zeros((n, n))
    for x in range(n):
        ix, jx = x // grid_size, x % grid_size
        for y in range(x + 1, n):
            iy, jy = y // grid_size, y % grid_size
            dist = abs(ix - iy) + abs(jx - jy)
            val = np.exp(-decay * dist) * scale
            affinity[x, y] = val
            affinity[y, x] = val
    return affinity


def compute_initial_organs(num_robots):
    """Initialize with full-connection organs: 1 organ covering all 25 voxels."""
    organs = torch.zeros(num_robots, 1, 25)
    organs[:, 0, :] = 1.0
    return organs


def run_solar_worker(
    idd, args, env_name, structure, organs, tc, saving_convention,
    transformer_mu, transformer_v, output_dir,
    ap_update_interval, affinity, verbose=True,
):
    """Single-robot SOLAR training with periodic AP organ updates."""
    print(f'Starting SOLAR training on \n{structure}\nat {saving_convention}...\n')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention is not None:
        log_dir = os.path.join(
            saving_convention[0], log_dir,
            "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])),
        )
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    num_gpus = max(1, torch.cuda.device_count()) if args.cuda else 1
    device = torch.device("cuda:{}".format(idd % num_gpus) if args.cuda else "cpu")

    envs = make_vec_envs(
        env_name, (structure.body, structure.connections),
        args.seed, args.num_processes, args.gamma, args.log_dir, device, False,
    )

    head_size = HEAD_SIZES[env_name]

    actor_critic = Policy(
        global_size=head_size, body_size=args.width,
        organs=organs, structure=structure,
        transformer_mu=transformer_mu, transformer_v=transformer_v,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param, args.ppo_epoch, args.num_mini_batch,
        args.value_loss_coef, args.entropy_coef,
        lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm,
    )

    rollouts = RolloutStorage(
        args.num_steps, args.num_processes,
        envs.observation_space.shape, envs.action_space, 100,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    from collections import deque
    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    max_determ_avg_reward = float('-inf')

    output_path = os.path.join(output_dir, "output{}.txt".format(saving_convention[1]))

    body_flat = structure.body.flatten()
    is_actuator = (body_flat == 3) | (body_flat == 4)
    actuator_indices = [b for b, bb in enumerate(is_actuator) if bb]

    for j in range(num_updates):
        if j > 0 and j % ap_update_interval == 0:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            base_reward = evaluate(
                actor_critic.organs, args.num_evals, actor_critic, obs_rms,
                env_name, (structure.body, structure.connections),
                args.seed, args.num_processes, eval_log_dir, device,
            )
            gaps = []
            for a_idx, a in enumerate(actuator_indices):
                ablation_reward = evaluate_ablation(
                    a_idx, actor_critic.organs, args.num_evals, actor_critic,
                    obs_rms, env_name, (structure.body, structure.connections),
                    args.seed, args.num_processes, eval_log_dir, device,
                )
                gaps.append(abs(base_reward - ablation_reward))

            preference = [np.median(gaps)] * 25
            for a_idx, b in enumerate(actuator_indices):
                preference[b] = gaps[a_idx]

            ap = AffinityPropagation(
                preference=preference, affinity="precomputed",
                random_state=args.seed, verbose=False,
            ).fit(affinity)
            labels = ap.labels_
            num_clusters = int(np.max(labels)) + 1
            new_organs = np.zeros((num_clusters, 25))
            for c in range(num_clusters):
                new_organs[c, labels == c] = 1
            actor_critic.organs = torch.tensor(new_organs, dtype=torch.float32)
            print("Organs updated at iter {} for robot {}: {} clusters".format(
                j, saving_convention[1], num_clusters))

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], actor_critic.organs)
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], actor_critic.organs).detach()

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma,
            args.gae_lambda, args.use_proper_time_limits,
        )

        if verbose:
            print("updating SOLAR policy for robot ", str(structure.label))
        value_loss, action_loss, dist_entropy = agent.update(rollouts, actor_critic.organs)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1 and verbose:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {} \n "
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n".format(
                    j, total_num_steps,
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards)))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(
                actor_critic.organs, args.num_evals, actor_critic, obs_rms,
                env_name, (structure.body, structure.connections),
                args.seed, args.num_processes, eval_log_dir, device,
            )
            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward
                temp_path = os.path.join(
                    saving_convention[0],
                    "task-{}_robot-{}_controller.pt".format(
                        structure.task_id, str(saving_convention[1])),
                )
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None),
                ], temp_path)
            with open(output_path, "a") as f:
                f.write("{}\t\t{}\t\t{}\n".format(
                    saving_convention[1], j, determ_avg_reward))

        if tc is not None:
            if tc(j):
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward

    return max_determ_avg_reward


if __name__ == "__main__":
    ctrl_args = parse_control_args()

    task_short = ctrl_args.task.split("-")[0].lower()
    if ctrl_args.output_name is None:
        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        exp_name = "{}-solar-{}".format(task_short, timestamp)
    else:
        exp_name = ctrl_args.output_name

    torch.multiprocessing.set_start_method("spawn")
    random.seed(ctrl_args.seed)
    np.random.seed(ctrl_args.seed)

    ppo_args = get_args(argv=[])
    ppo_args.train_iters = ctrl_args.train_iters
    ppo_args.lr = ctrl_args.lr
    ppo_args.num_processes = ctrl_args.num_processes
    ppo_args.num_steps = ctrl_args.num_steps
    ppo_args.num_cores = ctrl_args.num_cores
    ppo_args.width = ctrl_args.width
    ppo_args.eval_interval = ctrl_args.eval_interval
    ppo_args.log_interval = ctrl_args.log_interval
    ppo_args.cuda = not ctrl_args.no_cuda
    ppo_args.no_cuda = ctrl_args.no_cuda
    ppo_args.action_dim = ctrl_args.action_dim
    ppo_args.obs_dim = ctrl_args.obs_dim
    ppo_args.hidden_dim = ctrl_args.hidden_dim
    ppo_args.n_head = ctrl_args.n_head
    ppo_args.dropout = ctrl_args.dropout
    ppo_args.seed = ctrl_args.seed
    ppo_args.task = ctrl_args.task

    if ctrl_args.h_dim is not None:
        ppo_args.h_dim = ctrl_args.h_dim
    else:
        ppo_args.h_dim = HEAD_SIZES[ctrl_args.task]

    tc = TerminationCondition(ppo_args.train_iters)

    save_dir = os.path.join(curr_dir, "saved_data", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "controller"), exist_ok=True)
    temp_path = os.path.join(save_dir, "output.txt")
    output_dir = os.path.join(save_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    robos = torch.load(os.path.join(ctrl_args.morph_dir, "morph.pt"))
    indices = np.random.choice(robos.shape[0], size=ctrl_args.num_robots, replace=False)

    structures = []
    for label in indices:
        body = np.array(robos[label])
        conn = get_full_connectivity(body)
        structures.append(Structure(body, conn, label, task_id=0))
    print("Structures loaded: {} robots.".format(len(structures)))

    organs_list = compute_initial_organs(ctrl_args.num_robots)
    affinity = build_position_affinity(
        grid_size=ctrl_args.width,
        decay=ctrl_args.ap_affinity_decay,
        scale=ctrl_args.ap_affinity_scale,
    )

    head_size = ppo_args.h_dim
    transformer_mu = ATTBase(
        obs_dim=ppo_args.obs_dim, h_dim=head_size,
        action_dim=ppo_args.action_dim, hidden_dim=ppo_args.hidden_dim,
        body_size=ppo_args.width, n_head=ppo_args.n_head,
        dropout=ppo_args.dropout,
    )
    transformer_v = ATTBase(
        obs_dim=ppo_args.obs_dim, h_dim=head_size,
        action_dim=1, hidden_dim=ppo_args.hidden_dim,
        body_size=ppo_args.width, n_head=ppo_args.n_head,
        dropout=ppo_args.dropout,
    )

    save_path_controller = os.path.join(save_dir, "controller")
    group = mp.Group()
    for idx, structure in enumerate(structures):
        organ = organs_list[idx].squeeze(0) if organs_list[idx].dim() == 3 else organs_list[idx]
        job_args = (
            idx, ppo_args, ppo_args.task, structure, organ, tc,
            (save_path_controller, structure.label),
            transformer_mu, transformer_v, output_dir,
            ctrl_args.ap_update_interval, affinity,
        )
        group.add_job(run_solar_worker, job_args, callback=structure.set_reward)

    group.run_jobs(ppo_args.num_cores)

    for structure in structures:
        structure.compute_fitness()

    with open(temp_path, "a") as f:
        for structure in structures:
            f.write("{}\t\t{}\n".format(structure.label, structure.fitness))

    print("SOLAR experiment finished.")
