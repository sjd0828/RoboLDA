"""One-hot vector ↔ 5×5 morphology grid (EvoGym voxel classes 0–4)."""

from __future__ import annotations

import numpy as np
import torch


def operator(width: int) -> torch.Tensor:
    op = torch.zeros((5 * width**2, width**2))
    for i in range(width**2):
        op[5 * i : 5 * (i + 1), i] = torch.tensor([0, 1, 2, 3, 4], dtype=op.dtype)
    return op


def vec_to_morph(robots: torch.Tensor, operator_2: torch.Tensor, width: int) -> torch.Tensor:
    robots = torch.mm(robots, operator_2)
    return robots.reshape(robots.shape[0], width, width)


def morph_to_vec(robots: np.ndarray) -> torch.Tensor:
    bodies = robots.reshape(robots.shape[0], -1)
    out = []
    for i in range(bodies.shape[0]):
        body = []
        for j in range(bodies.shape[1]):
            voxel = [0, 0, 0, 0, 0]
            voxel[int(bodies[i, j])] = 1
            body += voxel
        out.append(body)
    return torch.tensor(np.array(out), dtype=torch.float32)


def onehot_flat_to_body(flat: torch.Tensor, width: int = 5) -> np.ndarray:
    """flat: (125,) or (B,125) float — argmax per cell -> int body."""
    t = flat.reshape(-1, width * width, 5)
    morph = t.argmax(dim=-1).cpu().numpy().astype(np.int64)
    return morph.reshape(-1, width, width)
