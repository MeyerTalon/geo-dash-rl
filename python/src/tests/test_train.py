import csv
import math
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.model.checkpoint import load_checkpoint
from src.model.config import EvalConfig, ModelConfig, PpoConfig
from src.model.network import ActorCritic
from src.model.ppo import EpochMetrics, UpdateMetrics
from src.run_log import RunLog, UpdateRow
from src.tests.test_rollout import MODEL_CONFIG, NUM_ENVS, ScriptedEnv
from src.train import (
    EpisodeHistory,
    StopRequest,
    TrainingSetup,
    check_run_is_consistent,
    mean_metrics_row,
    run_training,
)

OBS_SHAPE = (NUM_ENVS, 4, 18, 32)
PPO_CONFIG = PpoConfig(
    total_env_steps=32,
    rollout_steps=4,
    epochs_per_update=2,
    minibatch_size=4,
    learning_rate=0.001,
    adam_epsilon=1e-5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    checkpoint_interval_updates=2,
    metrics_window_episodes=5,
)
EVAL_CONFIG = EvalConfig(seed=100, episodes=4, interval_updates=2)
UPDATE_METRICS = UpdateMetrics(
    epochs=[EpochMetrics(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)], explained_variance=0.7
)


def row_for(episodes_finished: int, history: EpisodeHistory) -> UpdateRow:
    return mean_metrics_row(
        1, 8, 1.0, 8.0, 0.001, UPDATE_METRICS, episodes_finished, history
    )


def test_consistent_run_passes() -> None:
    check_run_is_consistent(MODEL_CONFIG, PPO_CONFIG, EVAL_CONFIG, NUM_ENVS, OBS_SHAPE)


@pytest.mark.parametrize(
    ('model', 'ppo', 'eval_config', 'match'),
    [
        (replace(MODEL_CONFIG, input_channels=1), PPO_CONFIG, EVAL_CONFIG, 'shape'),
        (replace(MODEL_CONFIG, input_height_px=9), PPO_CONFIG, EVAL_CONFIG, 'shape'),
        (replace(MODEL_CONFIG, input_width_px=9), PPO_CONFIG, EVAL_CONFIG, 'shape'),
        (
            replace(MODEL_CONFIG, action_count=3),
            PPO_CONFIG,
            EVAL_CONFIG,
            'action_count',
        ),
        (
            MODEL_CONFIG,
            replace(PPO_CONFIG, minibatch_size=3),
            EVAL_CONFIG,
            'minibatch_size',
        ),
        (
            MODEL_CONFIG,
            replace(PPO_CONFIG, total_env_steps=1),
            EVAL_CONFIG,
            'total_env_steps',
        ),
        (
            MODEL_CONFIG,
            PPO_CONFIG,
            EvalConfig(seed=100, episodes=3, interval_updates=2),
            'episodes',
        ),
    ],
)
def test_inconsistent_run_raises(
    model: ModelConfig, ppo: PpoConfig, eval_config: EvalConfig, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        check_run_is_consistent(model, ppo, eval_config, NUM_ENVS, OBS_SHAPE)


def train_with_fakes(tmp_path: Path, stop: StopRequest) -> tuple[Path, Path]:
    run_dir = tmp_path / 'runs'
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    env = ScriptedEnv()
    model = ActorCritic(MODEL_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_CONFIG.learning_rate)
    log = RunLog(run_dir)
    run_training(
        env,
        ScriptedEnv,
        model,
        optimizer,
        env.reset(),
        TrainingSetup(
            PPO_CONFIG,
            EVAL_CONFIG,
            NUM_ENVS,
            torch.device('cpu'),
            checkpoint_dir,
            {'seed': 0},
        ),
        log,
        stop,
    )
    log.close()
    return run_dir, checkpoint_dir


def test_training_writes_metrics_and_checkpoints(tmp_path: Path) -> None:
    run_dir, checkpoint_dir = train_with_fakes(tmp_path, StopRequest())

    with (run_dir / 'updates.csv').open(newline='') as csv_file:
        updates = list(csv.DictReader(csv_file))
    assert [row['env_steps'] for row in updates] == ['8', '16', '24', '32']
    assert math.isfinite(float(updates[0]['policy_loss']))
    assert float(updates[0]['max_progress']) == 1.0
    assert (run_dir / 'epochs.csv').read_text().count('\n') == 9
    assert (run_dir / 'eval.csv').read_text().count('\n') == 3
    assert sorted(item.name for item in checkpoint_dir.iterdir()) == [
        'best.pt',
        'latest.pt',
        'update_000002.pt',
        'update_000004.pt',
    ]
    assert load_checkpoint(checkpoint_dir / 'latest.pt').env_steps == 32


def test_stop_request_finishes_the_update_and_saves_latest(tmp_path: Path) -> None:
    stop = StopRequest()
    stop.request(signal.SIGINT, None)
    run_dir, checkpoint_dir = train_with_fakes(tmp_path, stop)

    assert (run_dir / 'updates.csv').read_text().count('\n') == 2
    assert load_checkpoint(checkpoint_dir / 'latest.pt').env_steps == 8


def test_update_without_finished_episodes_reports_the_recent_ones() -> None:
    history = EpisodeHistory(window_episodes=5)
    history.add([1.0, 3.0], [0.5, 1.0])
    row = row_for(episodes_finished=0, history=history)
    assert row.episodes_finished == 0
    assert row.mean_episode_return == 2.0
    assert row.mean_progress == 0.75
    assert row.max_progress == 1.0
    assert row.completion_rate == 0.5


def test_row_is_finite_before_any_episode_has_finished() -> None:
    row = row_for(episodes_finished=0, history=EpisodeHistory(window_episodes=5))
    assert all(math.isfinite(value) for value in row)
    assert row.mean_progress == 0.0


def test_history_keeps_only_the_latest_episodes() -> None:
    history = EpisodeHistory(window_episodes=2)
    history.add([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert list(history.progress_fractions) == [0.2, 0.3]
    assert list(history.returns) == [2.0, 3.0]
