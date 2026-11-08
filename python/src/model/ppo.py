from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.advantage import compute_gae
from src.model.config import PpoConfig
from src.model.network import ActorCritic

ADVANTAGE_STD_EPSILON = 1e-8


class RolloutBatch(NamedTuple):
    """`obs` is `(B, T, C, H, W)` uint8, `last_values` is `(B,)`, and the rest are `(B, T)`.
    B is the number of environments."""

    obs: Tensor
    actions: Tensor
    log_probs: Tensor
    values: Tensor
    rewards: Tensor
    dones: Tensor
    last_values: Tensor


class EpochMetrics(NamedTuple):
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float


class UpdateMetrics(NamedTuple):
    epochs: list[EpochMetrics]
    explained_variance: float


def clipped_policy_loss(
    log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, clip_epsilon: float
) -> Tensor:
    ratios = (log_probs - old_log_probs).exp()
    clipped_ratios = ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.min(ratios * advantages, clipped_ratios * advantages).mean()


def normalize_advantages(advantages: Tensor) -> Tensor:
    return (advantages - advantages.mean()) / (advantages.std() + ADVANTAGE_STD_EPSILON)


def explained_variance(values: Tensor, returns: Tensor) -> float:
    returns_variance = returns.var().item()
    if returns_variance == 0:
        return 0.0
    return 1 - (returns - values).var().item() / returns_variance


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PpoConfig,
    device: torch.device,
) -> UpdateMetrics:
    """The batch stays on the host, and each minibatch moves to `device` as uint8."""
    advantages = compute_gae(
        batch.rewards,
        batch.values,
        batch.dones,
        batch.last_values,
        config.gamma,
        config.gae_lambda,
    )
    returns = (advantages + batch.values).flatten()
    advantages = advantages.flatten()
    obs = batch.obs.flatten(0, 1)
    actions = batch.actions.flatten()
    old_log_probs = batch.log_probs.flatten()
    sample_count = obs.shape[0]

    epochs: list[EpochMetrics] = []
    for _ in range(config.epochs_per_update):
        minibatch_metrics: list[Tensor] = []
        shuffled_indices = torch.randperm(sample_count)
        for start in range(0, sample_count, config.minibatch_size):
            indices = shuffled_indices[start : start + config.minibatch_size]
            evaluation = model.evaluate_actions(
                obs[indices].to(device), actions[indices].to(device)
            )
            minibatch_old_log_probs = old_log_probs[indices].to(device)
            policy_loss = clipped_policy_loss(
                evaluation.log_probs,
                minibatch_old_log_probs,
                normalize_advantages(advantages[indices].to(device)),
                config.clip_epsilon,
            )
            value_loss = F.mse_loss(evaluation.values, returns[indices].to(device))
            entropy = evaluation.entropy.mean()
            loss = (
                policy_loss
                + config.value_loss_coef * value_loss
                - config.entropy_coef * entropy
            )
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            optimizer.step()

            log_ratios = (evaluation.log_probs - minibatch_old_log_probs).detach()
            ratios = log_ratios.exp()
            approx_kl = ((ratios - 1) - log_ratios).mean()
            clip_fraction = ((ratios - 1).abs() > config.clip_epsilon).float().mean()
            minibatch_metrics.append(
                torch.stack(
                    [
                        policy_loss.detach(),
                        value_loss.detach(),
                        entropy.detach(),
                        approx_kl,
                        clip_fraction,
                        grad_norm,
                    ]
                )
            )
        epochs.append(
            EpochMetrics(*torch.stack(minibatch_metrics).mean(dim=0).tolist())
        )
    return UpdateMetrics(
        epochs=epochs,
        explained_variance=explained_variance(batch.values.flatten(), returns),
    )
