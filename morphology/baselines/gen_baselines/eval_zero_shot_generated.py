"""
PPO-evaluate zero-shot generated morphologies on target tasks
(same pipeline as direct_transfer/run_direct_transfer.py).

Reads from ``gen_baselines/zero-shot robots/preset_{NN}_*/{morphvae,laser}/seed_*/gen_*.npz``,
running parallel evaluation for each of the 7 seeds.

Usage::

    python -m gen_baselines.eval_zero_shot_generated --preset 6
    python -m gen_baselines.eval_zero_shot_generated --preset 6 7 9
    python -m gen_baselines.eval_zero_shot_generated --preset 6 --zeros_root "./gen_baselines/zero-shot robots"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import List, Sequence

import numpy as np
import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
_BASELINES = os.path.dirname(_PKG)
_MORPHOLOGY = os.path.dirname(_BASELINES)
_EXTERNAL = os.path.join(_MORPHOLOGY, "externals", "pytorch_a2c_ppo_acktr_gail")
for _p in (_MORPHOLOGY, _EXTERNAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evogym.utils import get_full_connectivity
from ppo.arguments import get_args
from ppo.run_ppo import run_ppo
import utils.mp_group as mp
from utils.algo_utils import Structure, TerminationCondition

from .config import preset_by_index


def _load_structures_from_paths(paths, rel_root):
    structures = []
    source_labels = []
    for label, p in enumerate(paths):
        z = np.load(p)
        body = z["arr_0"]
        connections = get_full_connectivity(body)
        structures.append(Structure(body, connections, label, task_id=0))
        source_labels.append(os.path.relpath(p, rel_root))
    return structures, source_labels


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


def default_zeros_root() -> str:
    return os.path.join(_PKG, "zero-shot robots")


def find_preset_dir(zeros_root: str, preset_idx: int) -> str:
    prefix = f"preset_{preset_idx:02d}_"
    if not os.path.isdir(zeros_root):
        raise FileNotFoundError(f"zeros_root not found: {zeros_root}")
    cands = sorted(
        os.path.join(zeros_root, d) for d in os.listdir(zeros_root) if d.startswith(prefix)
    )
    if not cands:
        raise FileNotFoundError(f"No directory under {zeros_root!r} with prefix {prefix!r}")
    return cands[0]


def load_target_task(preset_dir: str, preset_idx: int) -> str:
    path = os.path.join(preset_dir, "preset_info.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        t = info.get("target_task")
        if t:
            return t
    return preset_by_index(preset_idx).target_task


def list_gen_npz(method_dir: str) -> List[str]:
    pat = os.path.join(method_dir, "gen_*.npz")
    return sorted(glob.glob(pat))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="PPO-evaluate morphvae & laser zero-shot gen (7 seeds) on target task."
    )
    p.add_argument(
        "--preset",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="One or more preset indices 1–9 (folders preset_NN_*). Example: --preset 6 7 9",
    )
    p.add_argument(
        "--zeros_root",
        type=str,
        default=None,
        help=f"Root of batch output (default: {default_zeros_root()})",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=("morphvae", "laser"),
        help="Subfolders to evaluate (default: morphvae laser).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="Seed subfolders seed_0 … (default: 0..6).",
    )
    p.add_argument("--train_iters", type=int, default=1000)
    p.add_argument("--num_cores", type=int, default=13, help="Parallel workers (same as run_direct_transfer).")
    p.add_argument("--ppo_seed", type=int, default=42, help="Fixed seed passed to get_args for PPO.")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output root under saved_data/ (default: auto timestamp).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    torch.multiprocessing.set_start_method("spawn", force=True)
    random.seed(args.ppo_seed)
    np.random.seed(args.ppo_seed)

    zeros_root = args.zeros_root or default_zeros_root()
    preset_indices = list(dict.fromkeys(args.preset))

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
            "eval_zero_shot_gen_" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
        )
    os.makedirs(exp_root, exist_ok=True)

    cfg_lines = [
        f"presets={preset_indices}",
        f"zeros_root={zeros_root}",
        f"methods={args.methods}",
        f"seeds={args.seeds}",
        f"train_iters={args.train_iters}",
        f"num_cores={args.num_cores}",
        f"ppo_seed={args.ppo_seed}",
    ]
    with open(os.path.join(exp_root, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cfg_lines) + "\n")

    summary: List[dict] = []

    for preset_idx in preset_indices:
        preset_dir = find_preset_dir(zeros_root, preset_idx)
        target_task = load_target_task(preset_dir, preset_idx)

        for method in args.methods:
            for seed in args.seeds:
                leaf = os.path.join(preset_dir, method, f"seed_{seed}")
                paths = list_gen_npz(leaf)
                if not paths:
                    print(f"[skip] no gen_*.npz under {leaf}", flush=True)
                    continue

                st, src_lbl = _load_structures_from_paths(paths, preset_dir)
                phase_name = f"p{preset_idx:02d}_{method}_seed{seed}"
                save_dir = os.path.join(exp_root, phase_name)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "source_npz_list.txt"), "w", encoding="utf-8") as f:
                    for pth in paths:
                        f.write(os.path.relpath(pth, preset_dir) + "\n")

                print(f"\n>>> {phase_name}  n={len(st)}  task={target_task}", flush=True)
                _run_ppo_phase(ppo_args, st, src_lbl, target_task, save_dir, tc, args.num_cores)

                fits = [s.fitness for s in st]
                summary.append(
                    {
                        "preset": preset_idx,
                        "phase": phase_name,
                        "target_task": target_task,
                        "method": method,
                        "gen_seed": seed,
                        "n": len(fits),
                        "mean": float(np.mean(fits)),
                        "max": float(np.max(fits)),
                        "median": float(np.median(fits)),
                    }
                )

    with open(os.path.join(exp_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"presets": preset_indices, "runs": summary}, f, indent=2)

    print(f"\nAll results under: {exp_root}", flush=True)


if __name__ == "__main__":
    main()
