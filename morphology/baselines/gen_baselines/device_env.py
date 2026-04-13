"""
Align CUDA device enumeration with nvidia-smi order (PCI bus).

Must be imported before any CUDA / torch.cuda initialization.
Default device: first visible GPU (cuda:0).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
