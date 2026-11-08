import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    """`rewards`, `values`, and `dones` are `(B, T)`, and `last_values` is `(B,)`. A true
    entry in `dones` ends the episode at that step, so nothing flows back across it."""
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(last_values)
    next_value = last_values
    for step in reversed(range(rewards.shape[1])):
        not_done = (~dones[:, step]).to(rewards.dtype)
        delta = rewards[:, step] + gamma * next_value * not_done - values[:, step]
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
        advantages[:, step] = next_advantage
        next_value = values[:, step]
    return advantages
