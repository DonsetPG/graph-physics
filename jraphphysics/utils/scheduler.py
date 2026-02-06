from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CosineWarmupScheduler:
    """
    Framework-agnostic cosine warmup scheduler helper.
    """

    warmup: int
    max_iters: int
    min_lr_factor: float = 0.001
    base_lrs: List[float] | None = None
    last_epoch: int = -1

    def get_lr(self) -> List[float]:
        lrs = self.base_lrs if self.base_lrs is not None else [1.0]
        factor = self.get_lr_factor(self.last_epoch)
        return [lr * factor for lr in lrs]

    def get_lr_factor(self, epoch: int) -> float:
        epoch = epoch + 1
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / max(self.warmup, 1)
        return float(max(lr_factor, self.min_lr_factor))
