from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn

TEMPORARY_SUFFIX = '.tmp'


class Checkpoint(NamedTuple):
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, object]
    config: dict[str, object]
    env_steps: int
    episode_count: int


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
    env_steps: int,
    episode_count: int,
) -> None:
    """Writes beside `path` and then replaces it, so an interrupted save leaves any
    existing checkpoint intact."""
    checkpoint = Checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        config=dict(config),
        env_steps=env_steps,
        episode_count=episode_count,
    )
    temporary_path = path.with_suffix(TEMPORARY_SUFFIX)
    torch.save(checkpoint._asdict(), temporary_path)
    temporary_path.replace(path)


def load_checkpoint(path: Path) -> Checkpoint:
    return Checkpoint(**torch.load(path, map_location='cpu', weights_only=True))
