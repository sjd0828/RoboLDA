"""Connectivity / actuator checks and adjacency (from gen_baselines/robogan/utils.py)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _is_in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _recursive_search(x: int, y: int, connectivity: np.ndarray, robot: np.ndarray) -> None:
    if robot[x][y] == 0:
        return
    if connectivity[x][y] != 0:
        return
    connectivity[x][y] = 1
    for x_offset in (-1, 1):
        nx, ny = x + x_offset, y
        if _is_in_bounds(nx, ny, robot.shape[1], robot.shape[0]):
            _recursive_search(nx, ny, connectivity, robot)
    for y_offset in (-1, 1):
        nx, ny = x, y + y_offset
        if _is_in_bounds(nx, ny, robot.shape[1], robot.shape[0]):
            _recursive_search(nx, ny, connectivity, robot)


def is_connected(robot: np.ndarray) -> bool:
    start = None
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] != 0:
                start = (i, j)
                break
        if start:
            break
    if start is None:
        return False
    connectivity = np.zeros(robot.shape)
    _recursive_search(start[0], start[1], connectivity, robot)
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] != 0 and connectivity[i][j] != 1:
                return False
    return True


def has_actuator(robot: np.ndarray) -> bool:
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] in (3, 4):
                return True
    return False


def get_full_connectivity(robot: np.ndarray) -> np.ndarray:
    out = []
    for i in range(robot.size):
        x = i % robot.shape[1]
        y = i // robot.shape[1]
        if robot[y][x] == 0:
            continue
        nx, ny = x + 1, y
        if _is_in_bounds(nx, ny, robot.shape[1], robot.shape[0]) and robot[ny][nx] != 0:
            out.append([x + robot.shape[1] * y, nx + robot.shape[1] * ny])
        nx, ny = x, y + 1
        if _is_in_bounds(nx, ny, robot.shape[1], robot.shape[0]) and robot[ny][nx] != 0:
            out.append([x + robot.shape[1] * y, nx + robot.shape[1] * ny])
    if len(out) == 0:
        return np.empty((0, 2)).T
    return np.array(out).T


def sample_valid_robot(shape: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Uniform-ish valid morphology (connected + ≥1 actuator)."""

    def draw(pd: np.ndarray) -> int:
        pd_copy = np.array(pd, dtype=np.float64)
        pd_copy = pd_copy / pd_copy.sum()
        r = float(rng.uniform(0.0, 1.0))
        s = 0.0
        for i in range(pd_copy.size):
            s += pd_copy[i]
            if r <= s:
                return i
        return pd_copy.size - 1

    while True:
        pd = np.ones(5) / 5.0
        pd[0] = 0.6
        robot = np.zeros(shape)
        for i in range(robot.shape[0]):
            for j in range(robot.shape[1]):
                robot[i][j] = draw(pd)
        if is_connected(robot) and has_actuator(robot):
            return robot.astype(np.int64)
