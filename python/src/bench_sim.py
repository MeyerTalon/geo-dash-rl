import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.config import load_config, require_int, require_section
from src.sim.config import SimConfig
from src.sim.vec_env import ACTION_COUNT, NativeVecEnv, VecEnv


class BenchResult(NamedTuple):
    env_steps: int
    elapsed_s: float
    episodes_finished: int
    mean_final_progress_fraction: float


def run_random_rollout(
    env: VecEnv, num_envs: int, steps: int, rng: np.random.Generator
) -> BenchResult:
    env.reset()
    final_progress_fractions: list[float] = []
    start_s = time.perf_counter()
    for _ in range(steps):
        actions = rng.integers(ACTION_COUNT, size=num_envs, dtype=np.uint8)
        batch = env.step(actions)
        final_progress_fractions.extend(batch.progress_fraction[batch.dones].tolist())
    elapsed_s = time.perf_counter() - start_s
    return BenchResult(
        env_steps=steps * num_envs,
        elapsed_s=elapsed_s,
        episodes_finished=len(final_progress_fractions),
        mean_final_progress_fraction=float(np.mean(final_progress_fractions))
        if final_progress_fractions
        else 0.0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = require_int(config, 'seed')
    sim_config = SimConfig.from_config(require_section(config, 'sim'))
    result = run_random_rollout(
        NativeVecEnv(sim_config, seed),
        sim_config.num_envs,
        args.steps,
        np.random.default_rng(seed),
    )
    print(
        f'{result.env_steps / result.elapsed_s:,.0f} env steps/s, '
        f'{result.episodes_finished} episodes, '
        f'mean final progress {result.mean_final_progress_fraction:.1%}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
