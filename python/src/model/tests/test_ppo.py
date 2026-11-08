import math

import pytest
import torch

from src.model.config import ModelConfig, PpoConfig
from src.model.network import ActorCritic
from src.model.ppo import (
    RolloutBatch,
    clipped_policy_loss,
    explained_variance,
    ppo_update,
)

CLIP_EPSILON = 0.2
NUM_ENVS = 2
ROLLOUT_STEPS = 4
MODEL_CONFIG = ModelConfig(
    input_channels=4,
    input_height_px=18,
    input_width_px=32,
    section_channels=(4,),
    residual_blocks_per_section=1,
    feature_dim=8,
    action_count=2,
)
PPO_CONFIG = PpoConfig(
    total_env_steps=100,
    rollout_steps=ROLLOUT_STEPS,
    epochs_per_update=2,
    minibatch_size=4,
    learning_rate=0.01,
    adam_epsilon=1e-5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=CLIP_EPSILON,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    checkpoint_interval_updates=1,
    metrics_window_episodes=5,
)


def loss_at_ratio(ratio: float, advantage: float) -> float:
    return clipped_policy_loss(
        torch.tensor([math.log(ratio)]),
        torch.tensor([0.0]),
        torch.tensor([advantage]),
        CLIP_EPSILON,
    ).item()


def test_ratio_inside_the_clip_range_is_used_directly() -> None:
    assert loss_at_ratio(1.1, 1.0) == pytest.approx(-1.1)


def test_positive_advantage_is_clipped_above() -> None:
    assert loss_at_ratio(2.0, 1.0) == pytest.approx(-1.2)


def test_negative_advantage_keeps_the_pessimistic_unclipped_ratio() -> None:
    assert loss_at_ratio(2.0, -1.0) == pytest.approx(2.0)


def test_explained_variance_is_one_for_perfect_values() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0])
    assert explained_variance(returns, returns) == pytest.approx(1.0)


def test_explained_variance_is_zero_for_constant_returns() -> None:
    assert explained_variance(torch.zeros(3), torch.ones(3)) == 0.0


def random_batch() -> RolloutBatch:
    shape = (NUM_ENVS, ROLLOUT_STEPS)
    frames = (*shape, 4, 18, 32)
    return RolloutBatch(
        obs=torch.randint(0, 256, frames, dtype=torch.uint8),
        actions=torch.randint(0, 2, shape),
        log_probs=torch.full(shape, math.log(0.5)),
        values=torch.zeros(shape),
        rewards=torch.randn(shape),
        dones=torch.zeros(shape, dtype=torch.bool),
        last_values=torch.zeros(NUM_ENVS),
    )


def test_update_changes_weights_and_reports_each_epoch() -> None:
    model = ActorCritic(MODEL_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_CONFIG.learning_rate)
    before = model.value_head.weight.detach().clone()
    metrics = ppo_update(
        model, optimizer, random_batch(), PPO_CONFIG, torch.device('cpu')
    )
    assert not torch.equal(before, model.value_head.weight)
    assert len(metrics.epochs) == PPO_CONFIG.epochs_per_update
    assert all(math.isfinite(value) for epoch in metrics.epochs for value in epoch)
