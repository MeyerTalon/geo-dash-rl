import numpy as np
from numpy.typing import NDArray

from src.bench_sim import run_random_rollout
from src.sim.vec_env import StepBatch

NUM_ENVS = 2
STEPS = 5


class DyingEnv:
    def reset(self) -> NDArray[np.uint8]:
        return np.zeros((NUM_ENVS, 1, 1), dtype=np.uint8)

    def step(self, actions: NDArray[np.uint8]) -> StepBatch:
        return StepBatch(
            obs=self.reset(),
            rewards=np.zeros(NUM_ENVS, dtype=np.float32),
            dones=np.array([True, False]),
            progress_fraction=np.array([0.5, 0.9], dtype=np.float32),
        )


def test_rollout_counts_steps_and_averages_only_finished_episodes() -> None:
    result = run_random_rollout(DyingEnv(), NUM_ENVS, STEPS, np.random.default_rng(0))
    assert result.env_steps == STEPS * NUM_ENVS
    assert result.episodes_finished == STEPS
    assert result.mean_final_progress_fraction == 0.5
