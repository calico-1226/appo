"""Proximal Policy Optimization (PPO) algorithm implementation."""

import torch

from appo.algorithms import PolicyGradient


class PPO(PolicyGradient):
    """Proximal Policy Optimization (PPO) algorithm."""
    def __init__(
            self,
            algo: str = 'ppo',
            clip: float = 0.2,
            **kwargs
    ):
        super().__init__(algo=algo, **kwargs)

        self.update_params_from_local(locals())
        self.clip = clip

    def compute_loss_pi(self, data: dict, **kwargs):
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        # Importance ratio
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        surrogate_adv = (torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()

        loss_pi = -surrogate_adv
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())

        return loss_pi, pi_info
