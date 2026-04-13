"""Conditional VAE on morphology (MorphVAE-style; PyTorch ELBO, stable vs. Pyro notebook)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from .. import device_env  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FLAT_DIM
from ..robot_utils import get_full_connectivity, has_actuator, is_connected
from ..progress_util import epoch_range, sample_pbar
from ..seed_utils import set_seed


class ConditionalMorphVAE(nn.Module):
    def __init__(self, num_tasks: int, latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        enc_in = FLAT_DIM + num_tasks
        self.enc = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        dec_in = latent_dim + num_tasks
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, FLAT_DIM),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        h = self.enc(torch.cat([x, y], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, y], dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        logits = self.decode(z, y)
        return logits, mu, logvar


def loss_elbo(logits: torch.Tensor, x_target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """x_target: (B,125) one-hot; logits -> (B,25,5) CE per cell."""
    b = x_target.shape[0]
    logit = logits.reshape(b, 25, 5)
    tgt = x_target.reshape(b, 25, 5)
    # cross entropy per voxel
    ce = F.cross_entropy(logit.reshape(-1, 5), tgt.argmax(dim=-1).reshape(-1), reduction="sum") / b
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return ce + kld


@dataclass
class MorphVAEConfig:
    epochs: int = 250
    batch_size: int = 64
    lr: float = 1e-3
    latent_dim: int = 64
    hidden: int = 256
    num_generate: int = 25
    max_attempts: int = 8000
    device: str = "cpu"


def run_morphvae(
    bodies: torch.Tensor,
    tasks: torch.Tensor,
    pretrain_tasks: List[str],
    target_task: str,
    out_dir: str,
    cfg: MorphVAEConfig,
    seed: int = 0,
    show_progress: bool = True,
) -> None:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)
    k_pre = len(pretrain_tasks)
    num_tasks = k_pre + 1
    n = bodies.shape[0]
    x = bodies.to(device)
    y = torch.zeros(n, num_tasks, device=device)
    y[:, :k_pre] = tasks.to(device)

    model = ConditionalMorphVAE(num_tasks, latent_dim=cfg.latent_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for _ in epoch_range(cfg.epochs, desc="morphvae train", disable=not show_progress):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if idx.numel() == 0:
                continue
            logits, mu, logvar = model(x[idx], y[idx])
            loss = loss_elbo(logits, x[idx], mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    y_t = torch.zeros(1, num_tasks, device=device)
    y_t[0, k_pre] = 1.0
    morphs: List[np.ndarray] = []
    seen = set()
    tries = 0
    sp = sample_pbar(cfg.num_generate, desc="morphvae sample", disable=not show_progress)
    while len(morphs) < cfg.num_generate and tries < cfg.max_attempts:
        tries += 1
        with torch.no_grad():
            z = torch.randn(1, cfg.latent_dim, device=device)
            logits = model.decode(z, y_t)
        morph = logits.reshape(25, 5).argmax(dim=-1).cpu().numpy().astype(np.int64).reshape(5, 5)
        key = morph.tobytes()
        if key in seen:
            sp.set_postfix(attempts=tries, ok=len(morphs))
            continue
        if not (is_connected(morph) and has_actuator(morph)):
            sp.set_postfix(attempts=tries, ok=len(morphs))
            continue
        seen.add(key)
        morphs.append(morph)
        sp.update(1)
        sp.set_postfix(attempts=tries, ok=len(morphs))
    sp.close()

    for i, m in enumerate(morphs):
        np.savez(os.path.join(out_dir, f"gen_{i:03d}.npz"), m, get_full_connectivity(m))

    meta = {
        "method": "morphvae",
        "seed": seed,
        "pretrain_tasks": pretrain_tasks,
        "target_task": target_task,
        "num_tasks_model": num_tasks,
        "target_slot_index": k_pre,
        "generated": len(morphs),
        "attempts": tries,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[morphvae] saved {len(morphs)}/{cfg.num_generate} bodies (attempts={tries}) -> {out_dir}",
        flush=True,
    )
