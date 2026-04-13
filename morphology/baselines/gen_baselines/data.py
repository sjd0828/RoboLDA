"""Load survivor .npz packs for one or more pre-training tasks."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch

from .config import ROBOT_SIZE, survivor_subdir
from .vec2morph import morph_to_vec


def default_survivors_root() -> str:
    pkg = os.path.dirname(os.path.abspath(__file__))
    examples = os.path.dirname(pkg)
    return os.path.join(examples, "task_similarity", "survivors")


def load_dataset(
    survivors_root: str,
    pretrain_tasks: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Returns
    -------
    bodies_onehot : (N, 125) float32
    tasks_onehot  : (N, K) float32  — K = len(pretrain_tasks)
    source_paths  : list of relative paths for logging
    """
    morphs: List[np.ndarray] = []
    task_indices: List[int] = []
    paths: List[str] = []

    k = len(pretrain_tasks)
    for t_idx, task_name in enumerate(pretrain_tasks):
        sub = survivor_subdir(task_name)
        task_dir = os.path.join(survivors_root, sub)
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(
                f"Survivor directory missing for {task_name}: {task_dir}\n"
                f"Expected layout: {survivors_root}/<{sub}>/*.npz"
            )
        files = sorted(f for f in os.listdir(task_dir) if f.endswith(".npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files in {task_dir}")
        for fn in files:
            p = os.path.join(task_dir, fn)
            z = np.load(p)
            body = z["arr_0"]
            if body.shape != (ROBOT_SIZE, ROBOT_SIZE):
                raise ValueError(f"{p}: expected body shape {(ROBOT_SIZE, ROBOT_SIZE)}, got {body.shape}")
            morphs.append(body.astype(np.int64))
            task_indices.append(t_idx)
            paths.append(os.path.join(sub, fn))

    bodies_t = morph_to_vec(np.stack(morphs, axis=0))
    oh = np.zeros((len(task_indices), k), dtype=np.float32)
    for i, ti in enumerate(task_indices):
        oh[i, ti] = 1.0
    tasks_t = torch.from_numpy(oh)
    return bodies_t, tasks_t, paths
