import numpy as np
import pytest
from numpy.typing import NDArray

from src.sim.config import SimConfig
from src.sim.vec_env import NativeVecEnv

MAX_STEPS = 2000

CONFIG = SimConfig(
    num_envs=3,
    frame_skip=4,
    obs_width_px=32,
    obs_height_px=18,
    level_length_units=100.0,
    death_penalty=1.0,
    completion_bonus=1.0,
)


def release_actions() -> NDArray[np.uint8]:
    return np.zeros(CONFIG.num_envs, dtype=np.uint8)


def test_reset_returns_one_uint8_frame_per_env() -> None:
    obs = NativeVecEnv(CONFIG, seed=0).reset()
    assert obs.shape == CONFIG.obs_shape
    assert obs.dtype == np.uint8


def test_step_returns_one_entry_per_env() -> None:
    env = NativeVecEnv(CONFIG, seed=0)
    env.reset()
    batch = env.step(release_actions())
    assert batch.obs.shape == CONFIG.obs_shape
    assert batch.rewards.shape == (CONFIG.num_envs,)
    assert batch.dones.dtype == np.bool_
    assert batch.progress_fraction.shape == (CONFIG.num_envs,)


def test_idling_envs_die_and_report_their_final_progress() -> None:
    env = NativeVecEnv(CONFIG, seed=0)
    env.reset()
    for _ in range(MAX_STEPS):
        batch = env.step(release_actions())
        if batch.dones.any():
            assert (batch.progress_fraction[batch.dones] > 0).all()
            assert (batch.rewards[batch.dones] < 0).all()
            return
    pytest.fail('no env finished an episode')


def test_same_seed_gives_the_same_frames() -> None:
    first = NativeVecEnv(CONFIG, seed=9)
    second = NativeVecEnv(CONFIG, seed=9)
    first.reset()
    second.reset()
    actions = np.array([0, 1, 0], dtype=np.uint8)
    np.testing.assert_array_equal(first.step(actions).obs, second.step(actions).obs)


def test_wrong_action_count_raises() -> None:
    env = NativeVecEnv(CONFIG, seed=0)
    with pytest.raises(ValueError, match='expected 3 actions'):
        env.step(np.zeros(CONFIG.num_envs + 1, dtype=np.uint8))


def test_unknown_action_index_raises() -> None:
    env = NativeVecEnv(CONFIG, seed=0)
    with pytest.raises(ValueError, match='action index 2'):
        env.step(np.full(CONFIG.num_envs, 2, dtype=np.uint8))
