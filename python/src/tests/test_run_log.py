import csv
from pathlib import Path

import pytest

from src.run_log import EpochRow, EvalRow, RunLog


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as csv_file:
        return list(csv.DictReader(csv_file))


def test_rows_land_in_their_own_csv_with_one_header(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    log = RunLog(run_dir)
    log.log_eval(EvalRow(1, 100, 0.5, 0.25, 4))
    log.log_eval(EvalRow(2, 200, 0.75, 0.5, 4))
    log.log_epoch(EpochRow(1, 100, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    log.close()

    eval_rows = read_rows(run_dir / 'eval.csv')
    assert [row['mean_progress'] for row in eval_rows] == ['0.5', '0.75']
    assert read_rows(run_dir / 'epochs.csv')[0]['grad_norm'] == '0.6'
    assert any(item.name.startswith('events.out') for item in run_dir.iterdir())


def test_existing_run_folder_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        RunLog(run_dir)
