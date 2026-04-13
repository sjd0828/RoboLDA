"""
Table 2 style: evaluate generation-0 EA zero-shot fitness on target tasks (pop=25, PPO training).

- **ga_bo**: random population via ``evogym.sample_robot`` (25 unique valid morphologies).
  GA/BO share the same random first generation; only generated and evaluated once.
- **cppn_gen0**: NEAT generation-0 random genomes decoded via CPPN into morphologies
  (same connectivity/actuator/dedup constraints), then PPO.

Defaults: presets 6-9, seeds 0-6 (7 repeats).

Dependency: CPPN branch requires ``neat-python``; not needed with ``--skip_cppn``.

Usage::

    python -m gen_baselines.eval_ea_gen0_baselines
    python -m gen_baselines.eval_ea_gen0_baselines --presets 6 7 --seeds 0 1 --num_cores 13
    python -m gen_baselines.eval_ea_gen0_baselines --skip_cppn
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
_BASELINES = os.path.dirname(_PKG)
_MORPHOLOGY = os.path.dirname(_BASELINES)
_EXTERNAL_PPO = os.path.join(_MORPHOLOGY, "externals", "pytorch_a2c_ppo_acktr_gail")
for _p in (_MORPHOLOGY, _EXTERNAL_PPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evogym import hashable, sample_robot
from ppo.arguments import get_args
from ppo.run_ppo import run_ppo
import utils.mp_group as mp
from utils.algo_utils import Structure, TerminationCondition

from .config import preset_by_index

_CPPN_DIR = os.path.join(_MORPHOLOGY, "baselines", "cppn_neat")


def _run_ppo_phase(args, structures, source_labels, target_task, save_dir, tc, num_cores):
    save_path_controller = os.path.join(save_dir, "controller")
    os.makedirs(save_path_controller, exist_ok=True)
    group = mp.Group()
    for structure in structures:
        ppo_args = (
            structure.label % 2, args, target_task, structure,
            tc, (save_path_controller, structure.label),
        )
        group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
    group.run_jobs(num_cores)
    for s in structures:
        s.compute_fitness()
    with open(os.path.join(save_dir, "output.txt"), "w", encoding="utf-8") as f:
        f.write("label\tsource_npz\tfitness\n")
        for s, src in zip(structures, source_labels):
            f.write(f"{s.label}\t{src}\t{s.fitness}\n")
    fits = [s.fitness for s in structures]
    print(f"\n[{target_task}] n={len(structures)}  mean={np.mean(fits):.4f}  "
          f"max={np.max(fits):.4f}  median={np.median(fits):.4f}")
    print(f"  saved: {save_dir}")


def sample_ga_style_population(
    pop_size: int,
    structure_shape: Tuple[int, int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Same as GA first generation: sample_robot + hash dedup."""
    seen: Dict = {}
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    while len(out) < pop_size:
        body, conn = sample_robot(structure_shape)
        h = hashable(body)
        if h in seen:
            continue
        seen[h] = True
        out.append((body, conn))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Table2-style gen-0 EA baselines: ga_bo (shared random pop) + cppn_neat gen0."
    )
    p.add_argument(
        "--presets",
        type=int,
        nargs="+",
        default=[6, 7, 8, 9],
        help="Preset indices (default: 6 7 8 9). Target task from config.preset_by_index.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="Random seeds for morphology sampling / NEAT init (default: 0..6).",
    )
    p.add_argument("--pop_size", type=int, default=25, help="First-generation population size (paper: 25).")
    p.add_argument("--structure_shape", type=int, nargs=2, default=[5, 5], metavar=("H", "W"))
    p.add_argument("--train_iters", type=int, default=1000)
    p.add_argument("--num_cores", type=int, default=13)
    p.add_argument(
        "--ppo_seed",
        type=int,
        default=42,
        help="Seed stored in PPO args (same as run_direct_transfer override).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Subfolder under saved_data/ (default: auto timestamp).",
    )
    p.add_argument("--skip_ga_bo", action="store_true", help="Only run cppn_gen0.")
    p.add_argument("--skip_cppn", action="store_true", help="Only run ga_bo.")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.skip_ga_bo and args.skip_cppn:
        raise SystemExit("Cannot set both --skip_ga_bo and --skip_cppn.")

    torch.multiprocessing.set_start_method("spawn", force=True)

    shape = (int(args.structure_shape[0]), int(args.structure_shape[1]))
    preset_indices = list(dict.fromkeys(args.presets))

    ppo_args = get_args([])
    ppo_args.num_processes = 4
    ppo_args.cuda = False
    ppo_args.no_cuda = True
    ppo_args.eval_interval = 50
    ppo_args.seed = args.ppo_seed

    tc = TerminationCondition(args.train_iters)

    if args.out_dir:
        exp_root = os.path.join(_MORPHOLOGY, "saved_data", args.out_dir)
    else:
        exp_root = os.path.join(
            _MORPHOLOGY,
            "saved_data",
            "ea_gen0_table2_" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
        )
    os.makedirs(exp_root, exist_ok=True)

    cfg_dump = {
        "presets": preset_indices,
        "seeds": args.seeds,
        "pop_size": args.pop_size,
        "structure_shape": list(shape),
        "train_iters": args.train_iters,
        "num_cores": args.num_cores,
        "ppo_seed": args.ppo_seed,
        "skip_ga_bo": args.skip_ga_bo,
        "skip_cppn": args.skip_cppn,
        "note": "ga_bo uses sample_robot; shared for GA/BO Table2 row.",
    }
    with open(os.path.join(exp_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    summary: List[dict] = []

    cppn_mod = None
    if not args.skip_cppn:
        from . import _ea_gen0_cppn as cppn_mod

    neat_cfg_path = os.path.join(_CPPN_DIR, "neat.cfg")

    for preset_idx in preset_indices:
        target_task = preset_by_index(preset_idx).target_task

        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if not args.skip_ga_bo:
                phase = f"p{preset_idx:02d}_ga_bo_seed{seed}"
                save_dir = os.path.join(exp_root, phase)
                os.makedirs(save_dir, exist_ok=True)
                morphs = sample_ga_style_population(args.pop_size, shape)
                struct_dir = os.path.join(save_dir, "structures")
                os.makedirs(struct_dir, exist_ok=True)
                for i, (body, conn) in enumerate(morphs):
                    pth = os.path.join(struct_dir, f"{i}.npz")
                    np.savez(pth, body, conn)

                structures: List[Structure] = []
                labels: List[str] = []
                for i, (body, conn) in enumerate(morphs):
                    structures.append(Structure(body, conn, i, 0))
                    labels.append(f"ga_bo/{i}")

                print(f"\n>>> {phase}  n={len(structures)}  task={target_task}", flush=True)
                _run_ppo_phase(ppo_args, structures, labels, target_task, save_dir, tc, args.num_cores)
                fits = [s.fitness for s in structures]
                summary.append(
                    {
                        "preset": preset_idx,
                        "target_task": target_task,
                        "seed": seed,
                        "method": "ga_bo",
                        "phase": phase,
                        "n": len(fits),
                        "mean": float(np.mean(fits)),
                        "max": float(np.max(fits)),
                        "median": float(np.median(fits)),
                    }
                )

            if not args.skip_cppn:
                assert cppn_mod is not None
                phase = f"p{preset_idx:02d}_cppn_gen0_seed{seed}"
                save_dir = os.path.join(exp_root, phase)
                os.makedirs(save_dir, exist_ok=True)
                structures, labels = cppn_mod.build_cppn_gen0_structures(
                    neat_cfg_path,
                    args.pop_size,
                    shape,
                    save_dir,
                    target_task,
                    args.train_iters,
                )
                print(f"\n>>> {phase}  n={len(structures)}  task={target_task}", flush=True)
                _run_ppo_phase(ppo_args, structures, labels, target_task, save_dir, tc, args.num_cores)
                fits = [s.fitness for s in structures]
                summary.append(
                    {
                        "preset": preset_idx,
                        "target_task": target_task,
                        "seed": seed,
                        "method": "cppn_gen0",
                        "phase": phase,
                        "n": len(fits),
                        "mean": float(np.mean(fits)),
                        "max": float(np.max(fits)),
                        "median": float(np.median(fits)),
                    }
                )

    with open(os.path.join(exp_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": summary}, f, indent=2)

    print(f"\nDone. Results: {exp_root}", flush=True)


if __name__ == "__main__":
    main()
