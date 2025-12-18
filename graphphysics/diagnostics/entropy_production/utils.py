"""Utility helpers for the entropy-production diagnostics.

This subpackage is self-contained (no dependency on the main graphphysics utils),
so it can be imported safely from training/analysis code.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


def numpy_to_torch(
    x: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert a numpy array (or torch tensor) to a torch.Tensor.

    If ``x`` is already a torch tensor, it is moved/cast only if requested.
    """
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device=device)
        if dtype is not None:
            x = x.to(dtype=dtype)
        return x
    t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def torch_to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor (or array-like) to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
