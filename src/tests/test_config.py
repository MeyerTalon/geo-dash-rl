from pathlib import Path

import pytest

from src.config import (
    load_config,
    require_float,
    require_int,
    require_mapping,
    require_section,
)


def test_load_config_reads_nested_sections(tmp_path: Path) -> None:
    path = tmp_path / 'config.yml'
    path.write_text('seed: 3\nsim:\n  num_envs: 2\n')
    config = load_config(path)
    assert require_int(config, 'seed') == 3
    assert require_int(require_section(config, 'sim'), 'num_envs') == 2


def test_missing_key_raises_instead_of_defaulting() -> None:
    with pytest.raises(KeyError, match='missing config key'):
        require_int({}, 'seed')


def test_bool_is_not_accepted_as_int() -> None:
    with pytest.raises(TypeError):
        require_int({'seed': True}, 'seed')


def test_float_key_accepts_an_int_literal() -> None:
    assert require_float({'penalty': 1}, 'penalty') == 1.0


def test_non_numeric_float_key_raises() -> None:
    with pytest.raises(TypeError):
        require_float({'penalty': 'high'}, 'penalty')


def test_non_mapping_raises() -> None:
    with pytest.raises(TypeError):
        require_mapping([1, 2], name='sim')
