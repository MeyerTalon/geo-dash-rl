from pathlib import Path

import torch

from src.model.checkpoint import load_checkpoint, save_checkpoint
from src.model.config import ModelConfig
from src.model.network import ActorCritic

CONFIG = ModelConfig(
    input_channels=4,
    input_height_px=18,
    input_width_px=32,
    section_channels=(4,),
    residual_blocks_per_section=1,
    feature_dim=8,
    action_count=2,
)
RAW_CONFIG: dict[str, object] = {'seed': 1, 'model': {'feature_dim': 8, 'widths': [4]}}


def test_round_trip_rebuilds_a_model_with_identical_outputs(tmp_path: Path) -> None:
    path = tmp_path / 'latest.pt'
    model = ActorCritic(CONFIG)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, model, optimizer, RAW_CONFIG, env_steps=12, episode_count=3)

    checkpoint = load_checkpoint(path)
    rebuilt = ActorCritic(CONFIG)
    rebuilt.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)

    frames = torch.zeros(2, 4, 18, 32, dtype=torch.uint8)
    assert torch.equal(model(frames)[0], rebuilt(frames)[0])
    assert checkpoint.config == RAW_CONFIG
    assert checkpoint.env_steps == 12
    assert checkpoint.episode_count == 3


def test_save_replaces_an_existing_checkpoint_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'latest.pt'
    model = ActorCritic(CONFIG)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, model, optimizer, RAW_CONFIG, env_steps=1, episode_count=0)
    save_checkpoint(path, model, optimizer, RAW_CONFIG, env_steps=2, episode_count=0)
    assert load_checkpoint(path).env_steps == 2
    assert [item.name for item in tmp_path.iterdir()] == ['latest.pt']
