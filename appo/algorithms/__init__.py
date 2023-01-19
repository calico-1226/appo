"""APPO Algorithms."""

from appo.algorithms.policy_gradient import PolicyGradient
from appo.algorithms.ppo import PPO
from appo.algorithms.ppo_lag import PPOLagrangian
from appo.algorithms.appo import APPO

REGISTRY = {
    "ppo": PPO,
    "pdo": PPOLagrangian,
    "appo": APPO,
}
