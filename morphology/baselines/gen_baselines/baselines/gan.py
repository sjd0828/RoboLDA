"""Conditional GAN baseline (RoboGAN-style), ref. gen_baselines/robogan."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .. import device_env  # noqa: F401 — align CUDA order with nvidia-smi before first use

import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical

from ..robot_utils import get_full_connectivity, has_actuator, is_connected
from ..progress_util import epoch_range, sample_pbar
from ..seed_utils import set_seed
from ..vec2morph import operator, vec_to_morph


def _g_loss(logits: torch.Tensor, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    probs = logits.exp() / logits.exp().sum(dim=2, keepdim=True)
    probs = (probs * x).sum(dim=2).sum(dim=1)
    weights = y_pred / (y_pred.sum() + 1e-8)
    return -(weights * probs).sum()


class Generator(nn.Module):
    def __init__(self, num_tasks: int, z_dim: int = 128, hidden_dim: int = 128, robot_size: int = 5):
        super().__init__()
        self.robot_size = robot_size
        self.z_dim = z_dim
        self.task_emb_dim = 128
        self.task_emb = nn.Linear(num_tasks, self.task_emb_dim, bias=False)

        def block(in_f: int, out_f: int, normalize: bool = False, activation: bool = True):
            layers = [nn.Linear(in_f, out_f)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_f, 0.8))
            if activation:
                layers.append(nn.ReLU())
            return layers

        n_out = robot_size**2 * 5
        self.model = nn.Sequential(
            *block(self.task_emb_dim + z_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, n_out, activation=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        y_e = self.task_emb(y)
        logits = self.model(torch.cat((y_e, z), dim=1)).reshape(z.shape[0], self.robot_size**2, 5)
        dist = OneHotCategorical(logits=logits)
        return dist.sample(), logits


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int = 128, robot_size: int = 5, task_emb_dim: int = 128):
        super().__init__()
        self.task_emb_dim = task_emb_dim
        n_in = robot_size**2 * 5 + task_emb_dim

        def block(in_f: int, out_f: int, normalize: bool = True, activation: bool = True):
            layers = [nn.Linear(in_f, out_f)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_f, 0.8))
            if activation:
                layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *block(n_in, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, 1, normalize=True, activation=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, emb_layer: nn.Module):
        y_e = emb_layer(y)
        flat = x.reshape(x.shape[0], -1)
        return self.model(torch.cat((y_e, flat), dim=1)).reshape(x.shape[0])


@dataclass
class GANConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    z_dim: int = 128
    num_generate: int = 25
    max_attempts: int = 5000
    device: str = "cpu"


def run_robogan(
    bodies: torch.Tensor,
    tasks: torch.Tensor,
    pretrain_tasks: List[str],
    target_task: str,
    out_dir: str,
    cfg: GANConfig,
    seed: int = 0,
    show_progress: bool = True,
) -> None:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)
    n = bodies.shape[0]
    k_pre = len(pretrain_tasks)
    # Extra slot for zero-shot target conditioning
    num_tasks = k_pre + 1
    bodies = bodies.to(device)
    tasks_ext = torch.zeros(n, num_tasks, device=device)
    tasks_ext[:, :k_pre] = tasks.to(device)

    gen = Generator(num_tasks, z_dim=cfg.z_dim).to(device)
    disc = Discriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    bce = nn.BCELoss()
    op = operator(5).to(device)

    for _ep in epoch_range(cfg.epochs, desc="robogan train", disable=not show_progress):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if idx.numel() == 0:
                continue
            real_x = bodies[idx].reshape(idx.shape[0], 25, 5)
            real_y = tasks_ext[idx]
            bsz = idx.shape[0]
            z = torch.randn(bsz, cfg.z_dim, device=device)
            fake_oh, _ = gen(z, real_y)
            # --- D ---
            opt_d.zero_grad()
            real_scores = disc(real_x, real_y, gen.task_emb)
            fake_scores = disc(fake_oh.detach(), real_y, gen.task_emb)
            loss_d = 0.5 * (
                bce(real_scores, torch.ones_like(real_scores))
                + bce(fake_scores, torch.zeros_like(fake_scores))
            )
            loss_d.backward()
            opt_d.step()
            # --- G ---
            opt_g.zero_grad()
            y_pred = disc(fake_oh, real_y, gen.task_emb).detach()
            _, logits = gen(z, real_y)
            loss_g = _g_loss(logits, fake_oh, y_pred.unsqueeze(1))
            loss_g.backward()
            opt_g.step()

    # Generate for target (last task index)
    gen.eval()
    disc.eval()
    y_t = torch.zeros(1, num_tasks, device=device)
    y_t[0, k_pre] = 1.0
    morphs: List[np.ndarray] = []
    seen = set()
    attempts = 0
    sp = sample_pbar(cfg.num_generate, desc="robogan sample", disable=not show_progress)
    while len(morphs) < cfg.num_generate and attempts < cfg.max_attempts:
        attempts += 1
        z = torch.randn(1, cfg.z_dim, device=device)
        with torch.no_grad():
            oh, _ = gen(z, y_t)
        vec = oh.reshape(1, 125)
        morph = vec_to_morph(vec, op, 5)[0].detach().cpu().numpy().astype(np.int64)
        key = morph.tobytes()
        if key in seen:
            sp.set_postfix(attempts=attempts, ok=len(morphs))
            continue
        if not (is_connected(morph) and has_actuator(morph)):
            sp.set_postfix(attempts=attempts, ok=len(morphs))
            continue
        seen.add(key)
        morphs.append(morph)
        sp.update(1)
        sp.set_postfix(attempts=attempts, ok=len(morphs))
    sp.close()

    for i, m in enumerate(morphs):
        conn = get_full_connectivity(m)
        np.savez(os.path.join(out_dir, f"gen_{i:03d}.npz"), m, conn)

    meta = {
        "method": "robogan",
        "seed": seed,
        "pretrain_tasks": pretrain_tasks,
        "target_task": target_task,
        "num_tasks_model": num_tasks,
        "target_slot_index": k_pre,
        "generated": len(morphs),
        "attempts": attempts,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[robogan] saved {len(morphs)}/{cfg.num_generate} bodies (attempts={attempts}) -> {out_dir}",
        flush=True,
    )
