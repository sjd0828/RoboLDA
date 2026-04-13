"""
Unified CLI for four generative baselines (RoboGAN, MorphVAE, c-DM, LASeR).

Run from the package parent directory::

    python -m gen_baselines --method robogan --preset 6 --out_dir ./out_p6_gan
    python -m gen_baselines --method morphvae --pretrain Walker-v0 DownStepper-v0 --target UpStepper-v0
    python -m gen_baselines --method all --preset 7 --out_dir ./out_p7

Survivor layout matches ``task_similarity/compute_similarity.py`` (folders under ``task_similarity/survivors``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from . import device_env  # noqa: F401 — before torch: match nvidia-smi GPU order

import torch

from .baselines.cdm import CDMConfig, run_cdm
from .baselines.gan import GANConfig, run_robogan
from .baselines.laser import LaserConfig, run_laser
from .baselines.morphvae import MorphVAEConfig, run_morphvae
from .config import list_presets_text, resolve_tasks
from .data import default_survivors_root, load_dataset


def _auto_device(requested: Optional[str]) -> str:
    """Default GPU: first visible device (`cuda:0`), ordered like ``nvidia-smi`` (PCI_BUS_ID)."""
    if requested:
        if requested == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(
        description="Train/sample generative morphology baselines (paper Table 1 presets)."
    )
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=("robogan", "morphvae", "cdm", "laser", "all"),
        help="Generative baseline (all = run each method into subfolders).",
    )
    p.add_argument("--preset", type=int, default=None, help="Transfer preset index 1–9 (paper Table 1).")
    p.add_argument(
        "--pretrain",
        nargs="+",
        default=None,
        help="EvoGym task names for pre-training data (alternative to --preset).",
    )
    p.add_argument("--target", type=str, default=None, help="Target / new task name (with --pretrain).")
    p.add_argument(
        "--survivors_root",
        type=str,
        default=None,
        help="Root folder of .npz survivors (default: examples/task_similarity/survivors).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: saved_data/gen_baselines_<timestamp>/<method>).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda | cuda:0 | cuda:1 ... Default: first visible GPU (cuda:0) if available.",
    )
    p.add_argument("--num_generate", type=int, default=25)
    p.add_argument("--list_presets", action="store_true", help="Print nine presets and exit.")
    # training budget (shared defaults)
    p.add_argument("--gan_epochs", type=int, default=80)
    p.add_argument("--cdm_epochs", type=int, default=200)
    p.add_argument("--vae_epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=64)
    # LASeR
    p.add_argument(
        "--laser_fallback",
        type=str,
        default="random",
        choices=("random", "error"),
        help="Without OPENAI_API_KEY/--openai_api_key, 'random' samples valid morphologies; 'error' raises.",
    )
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    p.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="LASeR: API base URL, equivalent to OPENAI_BASE_URL env var (include /v1 or SDK-expected path).",
    )
    p.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="LASeR: API key, equivalent to OPENAI_API_KEY (do not commit; falls back to env var).",
    )
    p.add_argument(
        "--openai_temperature",
        type=float,
        default=0.5,
        help="LASeR: chat sampling temperature (default 0.5).",
    )
    p.add_argument(
        "--laser_morphs_per_request",
        type=int,
        default=12,
        help="LASeR: max matrices per API request (filtered for connectivity/actuator afterwards).",
    )
    p.add_argument(
        "--laser_max_llm_rounds",
        type=int,
        default=40,
        help="LASeR: max API request rounds (prevents infinite loop).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm training/sampling bars (still prints one line per method).",
    )
    a = p.parse_args(argv)

    if a.list_presets:
        print(list_presets_text())
        return

    pretrain, target = resolve_tasks(a.preset, a.pretrain, a.target)
    survivors_root = a.survivors_root or default_survivors_root()
    device = _auto_device(a.device)

    examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if a.out_dir is None:
        stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        base = os.path.join(examples_dir, "saved_data", f"gen_baselines_{stamp}")
    else:
        base = a.out_dir
    os.makedirs(base, exist_ok=True)

    cfg_dump = {
        "preset": a.preset,
        "pretrain_tasks": pretrain,
        "target_task": target,
        "survivors_root": os.path.abspath(survivors_root),
        "method": a.method,
        "seed": a.seed,
        "device": device,
        "openai_base_url": a.openai_base_url,
        "openai_api_key": "***" if a.openai_api_key else None,
        "openai_temperature": a.openai_temperature,
        "laser_morphs_per_request": a.laser_morphs_per_request,
        "laser_max_llm_rounds": a.laser_max_llm_rounds,
    }
    with open(os.path.join(base, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    methods = (
        ("robogan", run_robogan),
        ("morphvae", run_morphvae),
        ("cdm", run_cdm),
        ("laser", run_laser),
    )

    if a.method == "all":
        todo = methods
    else:
        todo = [(a.method, dict(methods)[a.method])]

    need_survivors = any(m != "laser" for m, _ in todo)
    bodies = tasks = None
    if need_survivors:
        bodies, tasks, paths = load_dataset(survivors_root, pretrain)
        with open(os.path.join(base, "training_sources.txt"), "w", encoding="utf-8") as f:
            f.write(f"# n={len(paths)} morphology samples\n")
            for line in paths:
                f.write(line + "\n")

    for name, fn in todo:
        sub = os.path.join(base, name)
        os.makedirs(sub, exist_ok=True)
        if name == "robogan":
            cfg = GANConfig(
                epochs=a.gan_epochs,
                batch_size=min(a.batch_size, 32),
                num_generate=a.num_generate,
                device=device,
            )
            assert bodies is not None and tasks is not None
            fn(bodies, tasks, pretrain, target, sub, cfg, a.seed, show_progress=not a.quiet)
        elif name == "morphvae":
            cfg = MorphVAEConfig(
                epochs=a.vae_epochs,
                batch_size=a.batch_size,
                num_generate=a.num_generate,
                device=device,
            )
            assert bodies is not None and tasks is not None
            fn(bodies, tasks, pretrain, target, sub, cfg, a.seed, show_progress=not a.quiet)
        elif name == "cdm":
            cfg = CDMConfig(
                epochs=a.cdm_epochs,
                batch_size=a.batch_size,
                num_generate=a.num_generate,
                device=device,
            )
            assert bodies is not None and tasks is not None
            fn(bodies, tasks, pretrain, target, sub, cfg, a.seed, show_progress=not a.quiet)
        elif name == "laser":
            cfg = LaserConfig(
                num_generate=a.num_generate,
                model=a.openai_model,
                laser_fallback=a.laser_fallback,
                base_url=a.openai_base_url,
                api_key=a.openai_api_key,
                temperature=a.openai_temperature,
                morphs_per_request=a.laser_morphs_per_request,
                max_llm_rounds=a.laser_max_llm_rounds,
            )
            fn(pretrain, target, sub, cfg, a.seed, show_progress=not a.quiet)

    print(f"Done. Root: {base}")


if __name__ == "__main__":
    main()
