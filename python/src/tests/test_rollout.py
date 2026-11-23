import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.model.config import EvalConfig, ModelConfig
from src.model.network import ActorCritic
from src.rollout import (
    collect_rollout,
    eval_sim_config,
    evaluate_policy,
    initial_state,
)
from src.sim.config import SimConfig
from src.sim.vec_env import StepBatch

NUM_ENVS = 2
FRAME_SHAPE = (NUM_ENVS, 4, 18, 32)
DEVICE = torch.device('cpu')
MODEL_CONFIG = ModelConfig(
    input_channels=4,
    input_height_px=18,
    input_width_px=32,
    section_channels=(4,),
    residual_blocks_per_section=1,
    feature_dim=8,
    action_count=2,
)


class ScriptedEnv:
    """Env 0 finishes every 2 steps with full progress, and env 1 every 3 steps with half.
    Each frame is filled with the step number, so tests can see which frame was stored."""

    def __init__(self) -> None:
        self.step_count = 0

    def frames(self) -> NDArray[np.uint8]:
        return np.full(FRAME_SHAPE, self.step_count, dtype=np.uint8)

    def reset(self) -> NDArray[np.uint8]:
        self.step_count = 0
        return self.frames()

    def step(self, actions: NDArray[np.uint8]) -> StepBatch:
        self.step_count += 1
        return StepBatch(
            obs=self.frames(),
            rewards=np.ones(NUM_ENVS, dtype=np.float32),
            dones=np.array([self.step_count % 2 == 0, self.step_count % 3 == 0]),
            progress_fraction=np.array([1.0, 0.5], dtype=np.float32),
        )


def test_rollout_is_batch_first_and_stores_the_frame_each_action_saw() -> None:
    env = ScriptedEnv()
    rollout = collect_rollout(
        env, ActorCritic(MODEL_CONFIG), initial_state(env.reset()), 3, DEVICE
    )
    assert rollout.batch.obs.shape == (NUM_ENVS, 3, 4, 18, 32)
    assert rollout.batch.rewards.shape == (NUM_ENVS, 3)
    assert rollout.batch.last_values.shape == (NUM_ENVS,)
    assert rollout.batch.obs[0, :, 0, 0, 0].tolist() == [0, 1, 2]
    assert rollout.batch.dones.tolist() == [[False, True, False], [False, False, True]]


def test_episode_returns_reset_at_dones_and_carry_across_rollouts() -> None:
    env = ScriptedEnv()
    model = ActorCritic(MODEL_CONFIG)
    first = collect_rollout(env, model, initial_state(env.reset()), 3, DEVICE)
    assert first.finished_returns == [2.0, 3.0]
    assert first.next_state.episode_returns.tolist() == [1.0, 0.0]
    second = collect_rollout(env, model, first.next_state, 1, DEVICE)
    assert second.finished_returns == [2.0]


def test_each_env_contributes_an_equal_share_of_eval_episodes() -> None:
    result = evaluate_policy(ScriptedEnv(), ActorCritic(MODEL_CONFIG), 3, DEVICE)
    assert result.episodes == 6
    assert result.mean_progress == pytest.approx(0.75)
    assert result.completion_rate == pytest.approx(0.5)


def test_evaluation_returns_the_model_to_training_mode() -> None:
    model = ActorCritic(MODEL_CONFIG)
    evaluate_policy(ScriptedEnv(), model, 1, DEVICE)
    assert model.training


def test_eval_env_uses_the_held_out_levels_and_keeps_the_rest_of_the_sim() -> None:
    sim_config = SimConfig(
        num_envs=NUM_ENVS,
        frame_skip=4,
        frame_stack=4,
        obs_width_px=32,
        obs_height_px=18,
        level_length_units=100.0,
        death_penalty=1.0,
        completion_bonus=1.0,
        level_paths=('levels/train.json',),
        imported_fraction=0.5,
    )
    eval_config = EvalConfig(
        seed=1, episodes=2, interval_updates=1, level_paths=('levels/held_out.json',)
    )
    result = eval_sim_config(sim_config, eval_config)
    assert result.level_paths == ('levels/held_out.json',)
    assert result.imported_fraction == 0.5
    assert result.num_envs == NUM_ENVS
