"""Progress bars: tqdm when available, else no-op."""

from __future__ import annotations

from typing import Any


class _NoopPBar:
    __slots__ = ()

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, **kwargs: Any) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass

    def close(self) -> None:
        pass


def epoch_range(n: int, *, desc: str, disable: bool):
    """Epoch iterator with optional tqdm."""
    if disable:
        return range(n)
    try:
        from tqdm import trange

        return trange(n, desc=desc, leave=False, dynamic_ncols=True)
    except ImportError:
        return range(n)


def sample_pbar(total: int, *, desc: str, disable: bool):
    """tqdm(total=...) for sampling loop, or noop."""
    if disable:
        return _NoopPBar()
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)
    except ImportError:
        return _NoopPBar()
