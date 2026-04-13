import torch
import numpy as np


def operator(width):
    op = torch.zeros((5 * width ** 2, width ** 2))
    for i in range(width ** 2):
        op[5 * i:5 * (i + 1), i] = torch.tensor([0, 1, 2, 3, 4])
    return op


def vec_to_morph(robots, operator_2, width):
    robots = torch.mm(robots, operator_2)
    return robots.reshape(robots.shape[0], width, width)


def morph_to_vec(robots):
    """Convert (N, W, W) or (W, W) int morphology array to (N, W*W*5) one-hot tensor."""
    if robots.ndim == 2:
        robots = robots[np.newaxis]
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
