"""Task ↔ survivor subdir mapping and the nine paper transfer presets."""

from __future__ import annotations

from typing import List, NamedTuple, Sequence, Tuple

# Matches task_similarity/compute_similarity.py
TASK_TO_DIR = {
    "Walker-v0": "walker",
    "Pusher-v0": "pusher",
    "Carrier-v0": "carrier",
    "BridgeWalker-v0": "bridgewalker",
    "DownStepper-v0": "downstepper",
    "UpStepper-v0": "upstepper",
    "Climber-v0": "climber",
    "Climber-v1": "climber-v1",
    "Climber-v2": "climber-v2",
    "PlatformJumper-v0": "platformjumper",
    "GapJumper-v0": "gapjumper",
}


class TransferPreset(NamedTuple):
    index: int
    pretrain_tasks: Tuple[str, ...]
    target_task: str


# Table 1 style: (index, pre-training task(s), new task)
PRESETS: Tuple[TransferPreset, ...] = (
    TransferPreset(1, ("Walker-v0", "Pusher-v0", "Carrier-v0"), "BridgeWalker-v0"),
    TransferPreset(2, ("BridgeWalker-v0",), "Walker-v0"),
    TransferPreset(3, ("Walker-v0", "Carrier-v0"), "Pusher-v0"),
    TransferPreset(4, ("Walker-v0", "Pusher-v0"), "Carrier-v0"),
    TransferPreset(5, ("Walker-v0", "UpStepper-v0"), "DownStepper-v0"),
    TransferPreset(6, ("Walker-v0", "DownStepper-v0"), "UpStepper-v0"),
    TransferPreset(7, ("Climber-v0",), "Climber-v1"),
    TransferPreset(8, ("Climber-v0",), "Climber-v2"),
    TransferPreset(9, ("PlatformJumper-v0",), "GapJumper-v0"),
)

ROBOT_SIZE = 5
NUM_VOXEL_TYPES = 5
FLAT_DIM = ROBOT_SIZE * ROBOT_SIZE * NUM_VOXEL_TYPES  # 125


def preset_by_index(idx: int) -> TransferPreset:
    for p in PRESETS:
        if p.index == idx:
            return p
    raise ValueError(f"Unknown preset index {idx}; use 1–9 or --list-presets")


def list_presets_text() -> str:
    lines = ["Preset  pretrain_tasks  ->  target_task"]
    for p in PRESETS:
        pre = ", ".join(p.pretrain_tasks)
        lines.append(f"  {p.index}     {pre}  ->  {p.target_task}")
    return "\n".join(lines)


def resolve_tasks(
    preset: int | None,
    pretrain: Sequence[str] | None,
    target: str | None,
) -> Tuple[List[str], str]:
    if preset is not None:
        p = preset_by_index(preset)
        return list(p.pretrain_tasks), p.target_task
    if pretrain and target:
        return list(pretrain), target
    raise ValueError("Specify either --preset N or both --pretrain ... and --target NAME")


def survivor_subdir(task_name: str) -> str:
    if task_name not in TASK_TO_DIR:
        raise KeyError(
            f"Unknown task {task_name!r}. Known: {sorted(TASK_TO_DIR.keys())}"
        )
    return TASK_TO_DIR[task_name]
