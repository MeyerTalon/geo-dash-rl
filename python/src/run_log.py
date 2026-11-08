import csv
from pathlib import Path
from typing import NamedTuple

from torch.utils.tensorboard import SummaryWriter

UPDATES_NAME = 'updates'
EPOCHS_NAME = 'epochs'
EVAL_NAME = 'eval'
CSV_SUFFIX = '.csv'
INDEX_FIELDS = frozenset({'update', 'env_steps', 'epoch'})
PROGRESS_CHART_TITLE = 'train_vs_held_out_progress'
MEAN_PROGRESS_FIELD = 'mean_progress'


class UpdateRow(NamedTuple):
    update: int
    env_steps: int
    wall_time_s: float
    env_steps_per_s: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    explained_variance: float
    learning_rate: float
    episodes_finished: int
    mean_episode_return: float
    mean_progress: float
    max_progress: float
    completion_rate: float


class EpochRow(NamedTuple):
    update: int
    env_steps: int
    epoch: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float


class EvalRow(NamedTuple):
    update: int
    env_steps: int
    mean_progress: float
    completion_rate: float
    episodes: int


class RunLog:
    """Writes each row to `<name>.csv` and TensorBoard before returning, so a crashed run
    keeps its data. Refuses an existing `run_dir`, so a run can't overwrite another."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True)
        self.run_dir = run_dir
        self.epochs_logged = 0
        self.writer = SummaryWriter(log_dir=str(run_dir))
        self.writer.add_custom_scalars_multilinechart(
            [
                f'{UPDATES_NAME}/{MEAN_PROGRESS_FIELD}',
                f'{EVAL_NAME}/{MEAN_PROGRESS_FIELD}',
            ],
            title=PROGRESS_CHART_TITLE,
        )

    def log_update(self, row: UpdateRow) -> None:
        self.write(UPDATES_NAME, row, row.env_steps)

    def log_epoch(self, row: EpochRow) -> None:
        self.epochs_logged += 1
        self.write(EPOCHS_NAME, row, self.epochs_logged)

    def log_eval(self, row: EvalRow) -> None:
        self.write(EVAL_NAME, row, row.env_steps)

    def write(self, name: str, row: UpdateRow | EpochRow | EvalRow, step: int) -> None:
        values = row._asdict()
        path = (self.run_dir / name).with_suffix(CSV_SUFFIX)
        is_new_file = not path.exists()
        with path.open('a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if is_new_file:
                writer.writerow(values)
            writer.writerow(values.values())
        for field, value in values.items():
            if field not in INDEX_FIELDS:
                self.writer.add_scalar(f'{name}/{field}', value, step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
