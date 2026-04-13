"""
LASeR-style LLM morph proposal + fallback.

Full LASeR (Song et al., ICLR 2025) uses LLMs for diversified design; here we expose an
optional OpenAI-compatible chat call. Without API credentials, use --laser_fallback random.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..progress_util import sample_pbar
from ..robot_utils import get_full_connectivity, has_actuator, is_connected, sample_valid_robot
from ..seed_utils import set_seed


@dataclass
class LaserConfig:
    num_generate: int = 25
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None  # overrides OPENAI_BASE_URL
    api_key: Optional[str] = None  # overrides OPENAI_API_KEY
    laser_fallback: str = "error"  # "random" | "error"
    temperature: float = 0.5
    """Sampling temperature for chat completions."""
    morphs_per_request: int = 12
    """Ask the LLM for up to this many 5×5 matrices per API call (then filter valid)."""
    max_llm_rounds: int = 40
    """Max API rounds; increase if many parses fail."""


def _effective_api_key(cfg: LaserConfig) -> Optional[str]:
    return (cfg.api_key or os.environ.get("OPENAI_API_KEY") or "").strip() or None


def _effective_base_url(cfg: LaserConfig) -> Optional[str]:
    u = cfg.base_url or os.environ.get("OPENAI_BASE_URL") or ""
    u = u.strip()
    return u.rstrip("/") or None


def _openai_client_kwargs(cfg: LaserConfig) -> dict:
    kw: dict = {}
    bu = _effective_base_url(cfg)
    if bu:
        kw["base_url"] = bu
    key = _effective_api_key(cfg)
    if key:
        kw["api_key"] = key
    return kw


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_matrix_5x5(obj: object) -> bool:
    if not isinstance(obj, list) or len(obj) != 5:
        return False
    for row in obj:
        if not isinstance(row, list) or len(row) != 5:
            return False
    return True


def _coerce_matrix(obj: object) -> Optional[np.ndarray]:
    if not _is_matrix_5x5(obj):
        return None
    try:
        m = np.array(obj, dtype=np.int64)
        if m.min() >= 0 and m.max() <= 4:
            return m
    except (TypeError, ValueError):
        pass
    return None


def _parse_matrix_loose(text: str) -> Optional[np.ndarray]:
    """Single 5×5 from JSON array or 25 digits 0–4 in order."""
    text = _strip_code_fence(text)
    try:
        data = json.loads(text)
        m = _coerce_matrix(data)
        if m is not None:
            return m
    except json.JSONDecodeError:
        pass
    nums = re.findall(r"\b[0-4]\b", text)
    if len(nums) >= 25:
        flat = [int(x) for x in nums[:25]]
        return np.array(flat, dtype=np.int64).reshape(5, 5)
    return None


def _parse_matrices_from_json(data: object) -> List[np.ndarray]:
    """Extract all valid 5×5 matrices from parsed JSON."""
    out: List[np.ndarray] = []

    if isinstance(data, dict):
        for key in ("morphologies", "robots", "bodies", "matrices", "designs"):
            if key in data:
                data = data[key]
                break
        else:
            # single matrix under common key
            for key in ("body", "morphology", "matrix", "grid"):
                if key in data:
                    m = _coerce_matrix(data[key])
                    if m is not None:
                        out.append(m)
                    return out

    if isinstance(data, list):
        if len(data) == 5 and _is_matrix_5x5(data):
            m = _coerce_matrix(data)
            if m is not None:
                out.append(m)
            return out
        for item in data:
            m = _coerce_matrix(item)
            if m is not None:
                out.append(m)
    return out


def _parse_matrices(text: str) -> List[np.ndarray]:
    """Parse one or many 5×5 bodies from LLM output."""
    raw = _strip_code_fence(text.strip())
    try:
        data = json.loads(raw)
        got = _parse_matrices_from_json(data)
        if got:
            return got
    except json.JSONDecodeError:
        pass
    one = _parse_matrix_loose(raw)
    return [one] if one is not None else []


def _llm_once(user_prompt: str, cfg: LaserConfig) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai package for LASeR: pip install openai") from e
    client = OpenAI(**_openai_client_kwargs(cfg))
    system = (
        "You output ONLY valid JSON, no markdown. "
        "Voxel ints 0–4: 0 empty, 1 rigid, 2 soft, 3 horizontal actuator, 4 vertical actuator. "
        "Prefer bodies that are connected and include at least one actuator (3 or 4). "
        'Return an object: {"morphologies": [ M1, M2, ... ]} where each Mi is a 5×5 JSON array of rows.'
    )
    r = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(cfg.temperature),
    )
    return r.choices[0].message.content or ""


def run_laser(
    pretrain_tasks: List[str],
    target_task: str,
    out_dir: str,
    cfg: LaserConfig,
    seed: int,
    show_progress: bool = True,
) -> None:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    morphs: List[np.ndarray] = []
    api_key = _effective_api_key(cfg)
    llm_rounds = 0

    if not api_key or cfg.laser_fallback == "random":
        if cfg.laser_fallback != "random" and not api_key:
            raise RuntimeError(
                "LASeR: set OPENAI_API_KEY / --openai_api_key or pass --laser_fallback random "
                "(uses stochastic valid morphologies, not LLM)."
            )
        seen_r = set()
        sp = sample_pbar(cfg.num_generate, desc="laser random", disable=not show_progress)
        tries = 0
        while len(morphs) < cfg.num_generate:
            tries += 1
            m = sample_valid_robot((5, 5), rng)
            k = m.tobytes()
            if k in seen_r:
                sp.set_postfix(tries=tries, ok=len(morphs))
                continue
            seen_r.add(k)
            morphs.append(m)
            sp.update(1)
            sp.set_postfix(tries=tries, ok=len(morphs))
        sp.close()
    else:
        seen: set[bytes] = set()
        sp = sample_pbar(cfg.num_generate, desc="laser llm", disable=not show_progress)
        pre_s = ", ".join(pretrain_tasks)
        while len(morphs) < cfg.num_generate and llm_rounds < cfg.max_llm_rounds:
            llm_rounds += 1
            need = cfg.num_generate - len(morphs)
            ask_n = min(cfg.morphs_per_request, max(need + 4, 6))
            user = (
                f"Task: EvoGym {target_task}. Related pre-training tasks: {pre_s}. "
                f"Produce exactly {ask_n} DISTINCT novel 5×5 voxel bodies as JSON. "
                f'Format: {{"morphologies": [ [[row],[row],...5 rows], ... {ask_n} matrices ]}}. '
                "Each matrix: 5 rows × 5 integers in 0–4. Vary shapes; keep bodies connected with ≥1 actuator."
            )
            sp.set_postfix(rounds=llm_rounds, ok=len(morphs), ask=ask_n)
            try:
                raw = _llm_once(user, cfg)
                candidates = _parse_matrices(raw)
            except Exception:
                candidates = []
            parsed_n = len(candidates)
            for m in candidates:
                if len(morphs) >= cfg.num_generate:
                    break
                key = m.tobytes()
                if key in seen:
                    continue
                if not (is_connected(m) and has_actuator(m)):
                    continue
                seen.add(key)
                morphs.append(m)
                sp.update(1)
            sp.set_postfix(rounds=llm_rounds, ok=len(morphs), parsed=parsed_n)
        sp.close()

    for i, m in enumerate(morphs[: cfg.num_generate]):
        np.savez(os.path.join(out_dir, f"gen_{i:03d}.npz"), m, get_full_connectivity(m))

    used_llm = bool(api_key) and cfg.laser_fallback != "random"
    meta = {
        "method": "laser",
        "seed": seed,
        "pretrain_tasks": pretrain_tasks,
        "target_task": target_task,
        "generated": min(len(morphs), cfg.num_generate),
        "fallback": "llm" if used_llm else cfg.laser_fallback,
        "openai_base_url_set": _effective_base_url(cfg) is not None,
        "temperature": cfg.temperature,
        "morphs_per_request": cfg.morphs_per_request,
        "llm_rounds": llm_rounds if used_llm else 0,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[laser] saved {meta['generated']}/{cfg.num_generate} bodies -> {out_dir}", flush=True)
