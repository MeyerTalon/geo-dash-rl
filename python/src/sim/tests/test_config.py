from pathlib import Path

import pytest

from src.config import load_config, require_section
from src.sim.config import SimConfig

VALID_SECTION: dict[str, object] = {
    'num_envs': 4,
    'frame_skip': 4,
    'frame_stack': 2,
    'obs_width_px': 32,
    'obs_height_px': 18,
    'level_length_units': 100.0,
    'death_penalty': 1.0,
    'completion_bonus': 1.0,
    'level_paths': ['levels/1.json'],
    'imported_fraction': 0.5,
}


def test_from_config_reads_every_key() -> None:
    config = SimConfig.from_config(VALID_SECTION)
    assert config.num_envs == 4
    assert config.obs_shape == (4, 2, 18, 32)
    assert config.level_paths == ('levels/1.json',)
    assert config.imported_fraction == 0.5


def test_missing_key_raises() -> None:
    section = {
        key: value for key, value in VALID_SECTION.items() if key != 'frame_skip'
    }
    with pytest.raises(KeyError):
        SimConfig.from_config(section)


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('num_envs', 0),
        ('frame_skip', 0),
        ('frame_skip', 256),
        ('frame_stack', 0),
        ('frame_stack', 256),
        ('obs_width_px', 0),
        ('obs_height_px', 65_536),
        ('level_length_units', 0.0),
        ('death_penalty', -1.0),
        ('completion_bonus', -1.0),
        ('imported_fraction', -0.1),
        ('imported_fraction', 1.1),
    ],
)
def test_out_of_range_value_raises(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=key):
        SimConfig.from_config({**VALID_SECTION, key: value})


@pytest.mark.parametrize(
    ('name', 'level_paths'),
    [('default.yml', ()), ('imported.yml', ('levels/1.json',))],
)
def test_shipped_run_configs_name_their_levels(
    name: str, level_paths: tuple[str, ...]
) -> None:
    config = load_config(Path('src/configs') / name)
    assert (
        SimConfig.from_config(require_section(config, 'sim')).level_paths == level_paths
    )
