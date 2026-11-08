from pathlib import Path

import pytest

from src.config import load_config, require_section
from src.model.config import EvalConfig, ModelConfig, PpoConfig

CONFIG_DIR = Path('src/configs')

MODEL_SECTION: dict[str, object] = {
    'input_channels': 4,
    'input_height_px': 18,
    'input_width_px': 32,
    'section_channels': [4, 8],
    'residual_blocks_per_section': 1,
    'feature_dim': 16,
    'action_count': 2,
}

PPO_SECTION: dict[str, object] = {
    'total_env_steps': 100,
    'rollout_steps': 8,
    'epochs_per_update': 2,
    'minibatch_size': 4,
    'learning_rate': 0.001,
    'adam_epsilon': 0.00001,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'checkpoint_interval_updates': 10,
    'metrics_window_episodes': 5,
}


@pytest.mark.parametrize('name', ['default.yml', 'impala_x2.yml'])
def test_shipped_run_configs_load(name: str) -> None:
    config = load_config(CONFIG_DIR / name)
    ModelConfig.from_config(require_section(config, 'model'))
    PpoConfig.from_config(require_section(config, 'ppo'))
    EvalConfig.from_config(require_section(config, 'eval'))


def test_x2_config_doubles_only_the_channels() -> None:
    narrow = load_config(CONFIG_DIR / 'default.yml')
    wide = load_config(CONFIG_DIR / 'impala_x2.yml')
    narrow_model = ModelConfig.from_config(require_section(narrow, 'model'))
    wide_model = ModelConfig.from_config(require_section(wide, 'model'))
    assert wide_model.section_channels == (32, 64, 64)
    assert wide_model.input_channels == narrow_model.input_channels


@pytest.mark.parametrize('key', list(MODEL_SECTION))
def test_model_config_requires_every_key(key: str) -> None:
    section = {name: value for name, value in MODEL_SECTION.items() if name != key}
    with pytest.raises(KeyError):
        ModelConfig.from_config(section)


@pytest.mark.parametrize('key', list(PPO_SECTION))
def test_ppo_config_requires_every_key(key: str) -> None:
    section = {name: value for name, value in PPO_SECTION.items() if name != key}
    with pytest.raises(KeyError):
        PpoConfig.from_config(section)


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('input_channels', 0),
        ('feature_dim', 0),
        ('action_count', 0),
        ('section_channels', []),
        ('section_channels', [8, 0]),
    ],
)
def test_model_config_rejects_out_of_range_values(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        ModelConfig.from_config({**MODEL_SECTION, key: value})


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('rollout_steps', 0),
        ('metrics_window_episodes', 0),
        ('learning_rate', 0.0),
        ('gamma', 1.5),
        ('gae_lambda', -0.1),
        ('clip_epsilon', 0.0),
        ('entropy_coef', -1.0),
    ],
)
def test_ppo_config_rejects_out_of_range_values(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=key):
        PpoConfig.from_config({**PPO_SECTION, key: value})


@pytest.mark.parametrize(
    ('key', 'value'), [('seed', -1), ('episodes', 0), ('interval_updates', 0)]
)
def test_eval_config_rejects_out_of_range_values(key: str, value: int) -> None:
    section = {'seed': 1, 'episodes': 4, 'interval_updates': 2, key: value}
    with pytest.raises(ValueError, match=key):
        EvalConfig.from_config(section)
