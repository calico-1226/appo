"""Lagrangian component."""


import torch
from torch import optim


class Lagrangian:
    """Lagrangian component."""

    def __init__(
        self,
        cost_limit: float,
        multiplier_init: float,
        multiplier_lr: float,
        multiplier_optimizer: str,
    ):
        self.cost_limit = cost_limit
        self.multiplier_lr = multiplier_lr
        self.multiplier_optimizer = multiplier_optimizer
        self.multiplier_init = max(multiplier_init, 1e-5)

        self.multiplier = torch.nn.Parameter(
            torch.as_tensor(self.multiplier_init), requires_grad=True
        )

        assert hasattr(
            optim, multiplier_optimizer
        ), f"Optimizer={multiplier_optimizer} not found in torch."
        self.multiplier_optimizer = getattr(optim, multiplier_optimizer)(
            [
                self.multiplier,
            ],
            lr=multiplier_lr,
        )

    def compute_multiplier_loss(self, ep_cost):
        """Penalty loss for Lagrange multiplier."""
        return -self.multiplier * (ep_cost - self.cost_limit)

    def update_multiplier(self, ep_cost):
        """Update Lagrange multiplier (lambda)."""
        self.multiplier_optimizer.zero_grad()
        multiplier_loss = self.compute_multiplier_loss(ep_cost)
        multiplier_loss.backward()
        self.multiplier_optimizer.step()
        self.multiplier.data.clamp_(0)
