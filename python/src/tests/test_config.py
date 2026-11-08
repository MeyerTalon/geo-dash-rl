from pathlib import Path

import pytest

from src.config import (
    load_config,
    require_float,
    require_int,
    require_int_list,
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


def test_string_section_loads_the_referenced_file(tmp_path: Path) -> None:
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model' / 'small.yml').write_text('width: 8\n')
    path = tmp_path / 'run.yml'
    path.write_text('seed: 1\nmodel: model/small.yml\n')
    assert require_int(require_section(load_config(path), 'model'), 'width') == 8


def test_missing_referenced_file_raises(tmp_path: Path) -> None:
    path = tmp_path / 'run.yml'
    path.write_text('model: model/missing.yml\n')
    with pytest.raises(FileNotFoundError):
        load_config(path)


def test_int_list_reads_every_item() -> None:
    assert require_int_list({'widths': [16, 32]}, 'widths') == [16, 32]


@pytest.mark.parametrize('value', [16, [16, 'wide'], [True]])
def test_int_list_rejects_wrong_types(value: object) -> None:
    with pytest.raises(TypeError):
        require_int_list({'widths': value}, 'widths')
