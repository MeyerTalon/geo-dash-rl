import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.sim.config import SimConfig
from src.sim.vec_env import NativeVecEnv

MAX_STEPS = 2_000

CONFIG = SimConfig(
    num_envs=3,
    frame_skip=4,
    frame_stack=2,
    obs_width_px=32,
    obs_height_px=18,
    level_length_units=100.0,
    death_penalty=1.0,
    completion_bonus=1.0,
    level_paths=(),
    imported_fraction=0.0,
)
SPIKE_FREE_LEVEL_LENGTH_UNITS = 5.0


def release_actions() -> NDArray[np.uint8]:
    return np.zeros(CONFIG.num_envs, dtype=np.uint8)


def test_reset_returns_one_uint8_stack_per_env() -> None:
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


def test_reset_fills_each_stack_with_the_first_frame() -> None:
    obs = NativeVecEnv(CONFIG, seed=0).reset()
    assert (obs[:, 0] == obs[:, 1]).all()


def test_step_moves_the_newest_frame_towards_the_oldest_slot() -> None:
    env = NativeVecEnv(CONFIG, seed=0)
    env.reset()
    previous = env.step(release_actions()).obs
    batch = env.step(release_actions())
    assert not batch.dones.any()
    assert (batch.obs[:, 0] == previous[:, 1]).all()


def test_finished_env_stack_holds_only_the_next_episode() -> None:
    env = NativeVecEnv(CONFIG, seed=0)
    env.reset()
    for _ in range(MAX_STEPS):
        batch = env.step(release_actions())
        if batch.dones.any():
            finished = batch.obs[batch.dones]
            assert (finished[:, 0] == finished[:, 1]).all()
            return
    pytest.fail('no env finished an episode')


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


def write_spike_free_level(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {'length_units': SPIKE_FREE_LEVEL_LENGTH_UNITS, 'spikes': [], 'solids': []}
        )
    )
    return path


def test_imported_level_replaces_generated_ones(tmp_path: Path) -> None:
    level_path = write_spike_free_level(tmp_path / 'level.json')
    config = replace(CONFIG, level_paths=(str(level_path),), imported_fraction=1.0)
    env = NativeVecEnv(config, seed=0)
    env.reset()
    for _ in range(MAX_STEPS):
        batch = env.step(release_actions())
        if batch.dones.all():
            assert (batch.progress_fraction == 1.0).all()
            assert (batch.rewards > 0).all()
            return
    pytest.fail('imported spike-free level never completed')


def test_missing_level_file_raises(tmp_path: Path) -> None:
    config = replace(
        CONFIG, level_paths=(str(tmp_path / 'missing.json'),), imported_fraction=1.0
    )
    with pytest.raises(ValueError, match='cannot read level'):
        NativeVecEnv(config, seed=0)


def test_malformed_level_file_raises(tmp_path: Path) -> None:
    level_path = tmp_path / 'level.json'
    level_path.write_text('{}')
    config = replace(CONFIG, level_paths=(str(level_path),), imported_fraction=1.0)
    with pytest.raises(ValueError, match='invalid level'):
        NativeVecEnv(config, seed=0)
