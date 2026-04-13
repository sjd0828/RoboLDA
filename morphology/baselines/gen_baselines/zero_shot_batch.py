"""
Batch zero-shot generation: multiple presets x baselines x seeds.

Defaults: presets 6-9, 4 baselines, 7 random seeds each,
generating 25 valid morphologies per run into ``zero-shot robots/``.

Usage::

    python -m gen_baselines.zero_shot_batch
    python -m gen_baselines.zero_shot_batch --presets 6 7 --seeds 10 20 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Sequence, Tuple

from . import device_env  # noqa: F401 — before torch: match nvidia-smi GPU order

import torch

from .baselines.cdm import CDMConfig, run_cdm
from .baselines.gan import GANConfig, run_robogan
from .baselines.laser import LaserConfig, run_laser
from .baselines.morphvae import MorphVAEConfig, run_morphvae
from .config import preset_by_index
from .data import default_survivors_root, load_dataset
from .run import _auto_device

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(PKG_DIR, "zero-shot robots")
DEFAULT_PRESETS = (6, 7, 8, 9)
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6)
METHODS: Tuple[str, ...] = ("robogan", "morphvae", "cdm", "laser")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Batch zero-shot robots: presets × methods × seeds.")
    p.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT,
        help=f"Output root (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--presets",
        type=int,
        nargs="+",
        default=list(DEFAULT_PRESETS),
        help="Preset indices 1-9 (default: 6 7 8 9).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0..6).",
    )
    p.add_argument("--survivors_root", type=str, default=None)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda | cuda:N; default: first visible GPU (cuda:0) if available.",
    )
    p.add_argument("--num_generate", type=int, default=25)
    p.add_argument("--gan_epochs", type=int, default=80)
    p.add_argument("--cdm_epochs", type=int, default=200)
    p.add_argument("--vae_epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--laser_fallback", type=str, default="random", choices=("random", "error"))
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    p.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="LASeR: same as OPENAI_BASE_URL env var.",
    )
    p.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="LASeR: same as OPENAI_API_KEY (do not commit to repo).",
    )
    p.add_argument("--openai_temperature", type=float, default=0.5)
    p.add_argument("--laser_morphs_per_request", type=int, default=12)
    p.add_argument("--laser_max_llm_rounds", type=int, default=40)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars (only show simple batch count lines).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    survivors_root = args.survivors_root or default_survivors_root()
    device = _auto_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "out_dir": os.path.abspath(args.out_dir),
        "presets": args.presets,
        "seeds": args.seeds,
        "methods": list(METHODS),
        "num_generate": args.num_generate,
        "device": device,
        "openai_base_url": args.openai_base_url,
        "openai_api_key": "***" if args.openai_api_key else None,
        "openai_temperature": args.openai_temperature,
        "laser_morphs_per_request": args.laser_morphs_per_request,
        "laser_max_llm_rounds": args.laser_max_llm_rounds,
    }
    with open(os.path.join(args.out_dir, "batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_runs = len(args.presets) * len(METHODS) * len(args.seeds)
    run_idx = 0
    t0 = time.perf_counter()
    outer_pbar = None
    if not args.quiet:
        try:
            from tqdm import tqdm

            outer_pbar = tqdm(
                total=total_runs,
                desc="batch",
                unit="run",
                dynamic_ncols=True,
                smoothing=0.05,
            )
        except ImportError:
            print(
                f"[batch] {total_runs} runs (install tqdm for a progress bar: pip install tqdm)",
                flush=True,
            )

    show_inner = not args.quiet

    for preset_idx in args.presets:
        pr = preset_by_index(preset_idx)
        pretrain = list(pr.pretrain_tasks)
        target = pr.target_task
        bodies, tasks, paths = load_dataset(survivors_root, pretrain)

        preset_dir = os.path.join(args.out_dir, f"preset_{preset_idx:02d}_{_safe_tag(target)}")
        os.makedirs(preset_dir, exist_ok=True)
        with open(os.path.join(preset_dir, "preset_info.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "preset_index": preset_idx,
                    "pretrain_tasks": pretrain,
                    "target_task": target,
                    "n_train_samples": len(paths),
                },
                f,
                indent=2,
            )
        with open(os.path.join(preset_dir, "training_sources.txt"), "w", encoding="utf-8") as f:
            f.write(f"# n={len(paths)}\n")
            for line in paths:
                f.write(line + "\n")

        for method in METHODS:
            for seed in args.seeds:
                leaf = os.path.join(preset_dir, method, f"seed_{seed}")
                os.makedirs(leaf, exist_ok=True)
                run_cfg = {
                    "preset": preset_idx,
                    "method": method,
                    "seed": seed,
                    "target_task": target,
                    "pretrain_tasks": pretrain,
                }
                with open(os.path.join(leaf, "run_config.json"), "w", encoding="utf-8") as f:
                    json.dump(run_cfg, f, indent=2)

                run_idx += 1
                elapsed = time.perf_counter() - t0
                if outer_pbar is not None:
                    outer_pbar.set_postfix(
                        preset=preset_idx,
                        method=method,
                        seed=seed,
                        tgt=target.split("-")[0][:10],
                        refresh=False,
                    )
                else:
                    eta = (elapsed / run_idx) * (total_runs - run_idx) if run_idx else 0
                    print(
                        f"[batch {run_idx}/{total_runs}] preset={preset_idx} {method} seed={seed} "
                        f"target={target} | elapsed={elapsed:.0f}s eta~{eta:.0f}s",
                        flush=True,
                    )

                if method == "robogan":
                    run_robogan(
                        bodies,
                        tasks,
                        pretrain,
                        target,
                        leaf,
                        GANConfig(
                            epochs=args.gan_epochs,
                            batch_size=min(args.batch_size, 32),
                            num_generate=args.num_generate,
                            device=device,
                        ),
                        seed=seed,
                        show_progress=show_inner,
                    )
                elif method == "morphvae":
                    run_morphvae(
                        bodies,
                        tasks,
                        pretrain,
                        target,
                        leaf,
                        MorphVAEConfig(
                            epochs=args.vae_epochs,
                            batch_size=args.batch_size,
                            num_generate=args.num_generate,
                            device=device,
                        ),
                        seed=seed,
                        show_progress=show_inner,
                    )
                elif method == "cdm":
                    run_cdm(
                        bodies,
                        tasks,
                        pretrain,
                        target,
                        leaf,
                        CDMConfig(
                            epochs=args.cdm_epochs,
                            batch_size=args.batch_size,
                            num_generate=args.num_generate,
                            device=device,
                        ),
                        seed=seed,
                        show_progress=show_inner,
                    )
                elif method == "laser":
                    run_laser(
                        pretrain,
                        target,
                        leaf,
                        LaserConfig(
                            num_generate=args.num_generate,
                            model=args.openai_model,
                            laser_fallback=args.laser_fallback,
                            base_url=args.openai_base_url,
                            api_key=args.openai_api_key,
                            temperature=args.openai_temperature,
                            morphs_per_request=args.laser_morphs_per_request,
                            max_llm_rounds=args.laser_max_llm_rounds,
                        ),
                        seed,
                        show_progress=show_inner,
                    )

                if outer_pbar is not None:
                    outer_pbar.update(1)

    if outer_pbar is not None:
        outer_pbar.close()

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    manifest["total_wall_s"] = round(time.perf_counter() - t0, 2)
    with open(os.path.join(args.out_dir, "batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nBatch finished. Root: {args.out_dir}")


def _safe_tag(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
