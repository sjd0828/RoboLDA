"""Conditional DDPM on flattened morphology (c-DM baseline, ref. gen_baselines/cDM-code.ipynb)."""

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


def ddpm_schedules(beta1: float, beta2: float, T: int, device: torch.device):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32, device=device) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }


class DDPM(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int, betas: tuple, n_T: int, device: torch.device):
        super().__init__()
        self.input_dim = input_dim
        self.n_T = n_T
        self.timeemb = nn.Linear(1, hidden_dim)
        self.fc1 = nn.Linear(input_dim + condition_dim + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T, device).items():
            self.register_buffer(k, v)

    def forward_train(self, x0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x0: (B, D) continuous relaxation of one-hot morph; c: (B, cond_dim)."""
        b = x0.shape[0]
        t = torch.randint(1, self.n_T + 1, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrtab = self.sqrtab[t].unsqueeze(1)
        sqrtmab = self.sqrtmab[t].unsqueeze(1)
        x_t = sqrtab * x0 + sqrtmab * noise
        t_emb = self.timeemb((t.float() / self.n_T).unsqueeze(1))
        h = torch.cat([x_t, c, t_emb], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        eps_pred = self.fc4(h)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, n_sample: int, c: torch.Tensor) -> torch.Tensor:
        """c: (n_sample, cond_dim)"""
        device = c.device
        x_i = torch.randn(n_sample, self.input_dim, device=device)
        for i in range(self.n_T, 0, -1):
            z = torch.randn_like(x_i) if i > 1 else torch.zeros_like(x_i)
            t_is = torch.full((n_sample, 1), float(i), device=device)
            t_emb = self.timeemb(t_is / self.n_T)
            h = torch.cat([x_i, c, t_emb], dim=1)
            h = F.relu(self.fc1(h))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            eps = self.fc4(h)
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i


@dataclass
class CDMConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    n_T: int = 100
    betas: tuple = (1e-4, 0.02)
    num_generate: int = 25
    max_attempts: int = 8000
    device: str = "cpu"


def run_cdm(
    bodies: torch.Tensor,
    tasks: torch.Tensor,
    pretrain_tasks: List[str],
    target_task: str,
    out_dir: str,
    cfg: CDMConfig,
    seed: int = 0,
    show_progress: bool = True,
) -> None:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)
    k_pre = len(pretrain_tasks)
    cond_dim = k_pre + 1
    n = bodies.shape[0]

    x0 = bodies.to(device)
    tasks_e = torch.zeros(n, cond_dim, device=device)
    tasks_e[:, :k_pre] = tasks.to(device)

    ddpm = DDPM(FLAT_DIM, cond_dim, 128, cfg.betas, cfg.n_T, device).to(device)
    opt = torch.optim.Adam(ddpm.parameters(), lr=cfg.lr)

    for _ in epoch_range(cfg.epochs, desc="cdm train", disable=not show_progress):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if idx.numel() == 0:
                continue
            loss = ddpm.forward_train(x0[idx], tasks_e[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    ddpm.eval()
    morphs: List[np.ndarray] = []
    seen = set()
    bsz = min(64, max(8, cfg.num_generate))
    batches = 0
    sp = sample_pbar(cfg.num_generate, desc="cdm sample", disable=not show_progress)
    while len(morphs) < cfg.num_generate and batches * bsz < cfg.max_attempts:
        y_batch = torch.zeros(bsz, cond_dim, device=device)
        y_batch[:, k_pre] = 1.0
        samp = ddpm.sample(bsz, y_batch)
        batches += 1
        for j in range(bsz):
            if len(morphs) >= cfg.num_generate:
                break
            logits = samp[j].reshape(25, 5)
            morph = logits.argmax(dim=-1).cpu().numpy().astype(np.int64).reshape(5, 5)
            key = morph.tobytes()
            if key in seen:
                sp.set_postfix(batches=batches, ok=len(morphs))
                continue
            if not (is_connected(morph) and has_actuator(morph)):
                sp.set_postfix(batches=batches, ok=len(morphs))
                continue
            seen.add(key)
            morphs.append(morph)
            sp.update(1)
            sp.set_postfix(batches=batches, ok=len(morphs))
    sp.close()

    for i, m in enumerate(morphs):
        np.savez(os.path.join(out_dir, f"gen_{i:03d}.npz"), m, get_full_connectivity(m))

    meta = {
        "method": "cdm",
        "seed": seed,
        "pretrain_tasks": pretrain_tasks,
        "target_task": target_task,
        "cond_dim": cond_dim,
        "target_slot_index": k_pre,
        "generated": len(morphs),
        "sample_batches": batches,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[cdm] saved {len(morphs)}/{cfg.num_generate} bodies (diffusion_batches={batches}) -> {out_dir}",
        flush=True,
    )
