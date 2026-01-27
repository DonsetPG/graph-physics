from typing import List

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine learning rate scheduler with a warmup period.

    This scheduler adjusts the learning rate using a cosine function after a warmup period.
    The learning rate is first increased linearly during the warmup period, and then
    follows a cosine decay to a minimum learning rate factor.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int,
        max_iters: int,
        min_lr_factor: float = 0.001,
        last_epoch: int = -1,
    ):
        """
        Initializes the CosineWarmupScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup (int): Number of warmup iterations.
            max_iters (int): Total number of iterations.
            min_lr_factor (float, optional): Minimum learning rate factor (relative to base_lr).
                Defaults to 0.001.
            last_epoch (int, optional): The index of the last epoch. Defaults to -1.
        """
        self.warmup = warmup
        self.max_iters = max_iters
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Computes the updated learning rates for all parameter groups.

        Returns:
            List[float]: A list of updated learning rates for each parameter group.
        """
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """
        Computes the scaling factor for the learning rate at a given iteration.

        Args:
            epoch (int): The current iteration or epoch.

        Returns:
            float: The scaling factor for the learning rate.
        """
        epoch += 1  # Adjust because last_epoch starts from -1

        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        lr_factor = max(lr_factor, self.min_lr_factor)
        return lr_factor
