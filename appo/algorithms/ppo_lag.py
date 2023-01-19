"""PPO Lagrangian algorithm implementation."""

import torch

from appo.components import Lagrangian
from appo.algorithms import PPO


class PPOLagrangian(PPO):
    """PPO Lagrangian algorithm."""

    def __init__(
        self,
        algo: str = "ppo_lagrangian",
        # Lagrangian parameters
        cost_limit: float = 25.0,
        multiplier_init: float = 1e-3,
        multiplier_lr: float = 0.2,
        multiplier_optimizer: str = "Adam",
        **kwargs
    ):  # pylint: disable=too-many-arguments
        super().__init__(algo=algo, **kwargs)
        # save parameters
        self.cost_limit = cost_limit
        self.multiplier_init = multiplier_init
        self.multiplier_lr = multiplier_lr
        self.multiplier_optimizer = multiplier_optimizer
        self.update_params_from_local(locals())
        # create lagrangian
        self.lagrangian = Lagrangian(
            cost_limit=cost_limit,
            multiplier_init=multiplier_init,
            multiplier_lr=multiplier_lr,
            multiplier_optimizer=multiplier_optimizer,
        )

    def compute_loss_pi(self, data: dict, **kwargs):
        """Compute policy loss."""
        # Policy loss
        dist, _log_p = self.ac.pi(data["obs"], data["act"])
        ratio = torch.exp(_log_p - data["log_p"])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)

        surrogate_adv = (
            torch.min(ratio * data["adv"], ratio_clip * data["adv"])
        ).mean()
        surrogate_c_adv = (
            torch.max(ratio * data["cost_adv"], ratio_clip * data["cost_adv"])
        ).mean()

        # Lagrangian loss
        multiplier = self.lagrangian.multiplier.item()
        loss_pi = (-surrogate_adv + multiplier * surrogate_c_adv) / (1 + multiplier)

        # Add entropy bonus
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = 0.5 * (data["log_p"] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats("EpCosts")[0]
        # First update Lagrange multiplier parameter
        self.lagrangian.update_multiplier(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.store(LagrangianMultiplier=self.lagrangian.multiplier.item())
