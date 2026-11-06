from collections.abc import Mapping
from pathlib import Path

import yaml


def load_config(path: Path) -> dict[str, object]:
    loaded: object = yaml.safe_load(path.read_text())
    return require_mapping(loaded, name=str(path))


def require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f'{name} must be a mapping, got {type(value).__name__}')
    return {str(key): item for key, item in value.items()}


def require_key(section: Mapping[str, object], key: str) -> object:
    if key not in section:
        raise KeyError(f'missing config key {key!r}')
    return section[key]


def require_section(section: Mapping[str, object], key: str) -> dict[str, object]:
    return require_mapping(require_key(section, key), name=key)


def require_int(section: Mapping[str, object], key: str) -> int:
    value = require_key(section, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'config key {key!r} must be an int, got {value!r}')
    return value


def require_float(section: Mapping[str, object], key: str) -> float:
    value = require_key(section, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f'config key {key!r} must be a number, got {value!r}')
    return float(value)
