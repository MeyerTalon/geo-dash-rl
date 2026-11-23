import argparse
import random
import signal
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from src.config import load_config, require_int, require_section
from src.model.checkpoint import save_checkpoint
from src.model.config import EvalConfig, ModelConfig, PpoConfig
from src.model.device import select_device
from src.model.network import ActorCritic
from src.model.ppo import EpochMetrics, UpdateMetrics, ppo_update
from src.rollout import (
    collect_rollout,
    completion_rate,
    eval_sim_config,
    evaluate_policy,
    initial_state,
)
from src.run_log import EpochRow, EvalRow, RunLog, UpdateRow
from src.sim.config import SimConfig
from src.sim.vec_env import ACTION_COUNT, NativeVecEnv, VecEnv

RUNS_DIR = Path('runs')
CHECKPOINTS_DIR = Path('checkpoints')
LATEST_CHECKPOINT_NAME = 'latest.pt'
BEST_CHECKPOINT_NAME = 'best.pt'


class StopRequest:
    """Set from a signal handler and read between updates, so a run only stops with its
    weights, optimizer, and step counts in step with each other."""

    def __init__(self) -> None:
        self.requested = False

    def request(self, signal_number: int, frame: FrameType | None) -> None:
        self.requested = True


class EpisodeHistory:
    """The most recent finished episodes. Levels have a fixed length, so once games survive
    they finish in bursts, and many updates finish no episode at all."""

    def __init__(self, window_episodes: int) -> None:
        self.returns: deque[float] = deque(maxlen=window_episodes)
        self.progress_fractions: deque[float] = deque(maxlen=window_episodes)

    def add(self, returns: list[float], progress_fractions: list[float]) -> None:
        self.returns.extend(returns)
        self.progress_fractions.extend(progress_fractions)


class TrainingSetup(NamedTuple):
    ppo: PpoConfig
    eval_config: EvalConfig
    num_envs: int
    device: torch.device
    checkpoint_dir: Path
    raw_config: Mapping[str, object]


def check_run_is_consistent(
    model: ModelConfig,
    ppo: PpoConfig,
    eval_config: EvalConfig,
    num_envs: int,
    obs_shape: tuple[int, ...],
) -> None:
    expected_shape = (
        num_envs,
        model.input_channels,
        model.input_height_px,
        model.input_width_px,
    )
    if obs_shape != expected_shape:
        raise ValueError(
            f'model expects observations of shape {expected_shape}, '
            f'but the environment returns {obs_shape}'
        )
    if model.action_count != ACTION_COUNT:
        raise ValueError(
            f'action_count is {model.action_count}, '
            f'but the environment has {ACTION_COUNT} actions'
        )
    samples_per_update = num_envs * ppo.rollout_steps
    if samples_per_update % ppo.minibatch_size != 0:
        raise ValueError(
            f'minibatch_size {ppo.minibatch_size} must divide '
            f'the {samples_per_update} samples in each update'
        )
    if ppo.total_env_steps < samples_per_update:
        raise ValueError(
            f'total_env_steps {ppo.total_env_steps} is less than '
            f'the {samples_per_update} steps in one update'
        )
    if eval_config.episodes % num_envs != 0:
        raise ValueError(
            f'eval episodes {eval_config.episodes} must divide evenly '
            f'across {num_envs} environments'
        )


def check_eval_levels_are_held_out(
    sim_config: SimConfig, eval_config: EvalConfig
) -> None:
    shared = set(sim_config.level_paths) & set(eval_config.level_paths)
    if shared:
        raise ValueError(
            f'levels in both the training pool and the held-out set: {sorted(shared)}'
        )


def mean_metrics_row(
    update: int,
    env_steps: int,
    wall_time_s: float,
    env_steps_per_s: float,
    learning_rate: float,
    metrics: UpdateMetrics,
    episodes_finished: int,
    history: EpisodeHistory,
) -> UpdateRow:
    """Losses are averaged over the update's epochs. Episode columns describe the episodes
    in `history`, so an update that finishes no episode still has a defined value."""
    means = EpochMetrics(*np.mean(metrics.epochs, axis=0).tolist())
    progress_fractions = list(history.progress_fractions)
    has_episodes = bool(progress_fractions)
    return UpdateRow(
        update,
        env_steps,
        wall_time_s,
        env_steps_per_s,
        *means,
        metrics.explained_variance,
        learning_rate,
        episodes_finished,
        float(np.mean(history.returns)) if has_episodes else 0.0,
        float(np.mean(progress_fractions)) if has_episodes else 0.0,
        max(progress_fractions) if has_episodes else 0.0,
        completion_rate(progress_fractions) if has_episodes else 0.0,
    )


def run_training(
    env: VecEnv,
    make_eval_env: Callable[[], VecEnv],
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    first_obs: NDArray[np.uint8],
    setup: TrainingSetup,
    log: RunLog,
    stop: StopRequest,
) -> None:
    ppo = setup.ppo
    samples_per_update = setup.num_envs * ppo.rollout_steps
    state = initial_state(first_obs)
    history = EpisodeHistory(ppo.metrics_window_episodes)
    env_steps = 0
    episode_count = 0
    best_progress = float('-inf')
    start_s = time.perf_counter()

    def save(name: str) -> None:
        save_checkpoint(
            setup.checkpoint_dir / name,
            model,
            optimizer,
            setup.raw_config,
            env_steps,
            episode_count,
        )

    for update in range(1, ppo.total_env_steps // samples_per_update + 1):
        update_start_s = time.perf_counter()
        rollout = collect_rollout(env, model, state, ppo.rollout_steps, setup.device)
        metrics = ppo_update(model, optimizer, rollout.batch, ppo, setup.device)
        state = rollout.next_state
        env_steps += samples_per_update
        episode_count += len(rollout.finished_progress_fractions)
        history.add(rollout.finished_returns, rollout.finished_progress_fractions)

        now_s = time.perf_counter()
        row = mean_metrics_row(
            update,
            env_steps,
            now_s - start_s,
            samples_per_update / (now_s - update_start_s),
            ppo.learning_rate,
            metrics,
            len(rollout.finished_progress_fractions),
            history,
        )
        log.log_update(row)
        for epoch, epoch_metrics in enumerate(metrics.epochs, start=1):
            log.log_epoch(EpochRow(update, env_steps, epoch, *epoch_metrics))
        print(
            f'update {update}: {env_steps:,} steps, '
            f'{row.env_steps_per_s:,.0f} steps/s, '
            f'progress {row.mean_progress:.1%}, '
            f'value loss {row.value_loss:.4f}'
        )

        if update % setup.eval_config.interval_updates == 0:
            result = evaluate_policy(
                make_eval_env(),
                model,
                setup.eval_config.episodes // setup.num_envs,
                setup.device,
            )
            log.log_eval(
                EvalRow(
                    update,
                    env_steps,
                    result.mean_progress,
                    result.completion_rate,
                    result.episodes,
                )
            )
            print(f'held-out progress {result.mean_progress:.1%}')
            if result.mean_progress > best_progress:
                best_progress = result.mean_progress
                save(BEST_CHECKPOINT_NAME)

        if update % ppo.checkpoint_interval_updates == 0:
            save(LATEST_CHECKPOINT_NAME)
            save(f'update_{update:06d}.pt')

        if stop.requested:
            print(f'stopping after update {update}')
            break
    save(LATEST_CHECKPOINT_NAME)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = require_int(config, 'seed')
    sim_config = SimConfig.from_config(require_section(config, 'sim'))
    model_config = ModelConfig.from_config(require_section(config, 'model'))
    ppo = PpoConfig.from_config(require_section(config, 'ppo'))
    eval_config = EvalConfig.from_config(require_section(config, 'eval'))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = NativeVecEnv(sim_config, seed)
    first_obs = env.reset()
    check_run_is_consistent(
        model_config, ppo, eval_config, sim_config.num_envs, first_obs.shape
    )
    check_eval_levels_are_held_out(sim_config, eval_config)

    checkpoint_dir = CHECKPOINTS_DIR / args.run_name
    checkpoint_dir.mkdir(parents=True)
    log = RunLog(RUNS_DIR / args.run_name)

    device = select_device()
    model = ActorCritic(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo.learning_rate, eps=ppo.adam_epsilon
    )
    stop = StopRequest()
    signal.signal(signal.SIGINT, stop.request)
    run_training(
        env,
        lambda: NativeVecEnv(
            eval_sim_config(sim_config, eval_config), eval_config.seed
        ),
        model,
        optimizer,
        first_obs,
        TrainingSetup(
            ppo, eval_config, sim_config.num_envs, device, checkpoint_dir, config
        ),
        log,
        stop,
    )
    log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
