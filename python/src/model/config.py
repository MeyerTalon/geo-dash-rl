from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from src.config import require_float, require_int, require_int_list


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    input_height_px: int
    input_width_px: int
    section_channels: tuple[int, ...]
    residual_blocks_per_section: int
    feature_dim: int
    action_count: int

    @classmethod
    def from_config(cls, section: Mapping[str, object]) -> Self:
        config = cls(
            input_channels=require_int(section, 'input_channels'),
            input_height_px=require_int(section, 'input_height_px'),
            input_width_px=require_int(section, 'input_width_px'),
            section_channels=tuple(require_int_list(section, 'section_channels')),
            residual_blocks_per_section=require_int(
                section, 'residual_blocks_per_section'
            ),
            feature_dim=require_int(section, 'feature_dim'),
            action_count=require_int(section, 'action_count'),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, value in (
            ('input_channels', self.input_channels),
            ('input_height_px', self.input_height_px),
            ('input_width_px', self.input_width_px),
            ('residual_blocks_per_section', self.residual_blocks_per_section),
            ('feature_dim', self.feature_dim),
            ('action_count', self.action_count),
        ):
            if value < 1:
                raise ValueError(f'{name} must be at least 1, got {value}')
        if not self.section_channels or min(self.section_channels) < 1:
            raise ValueError(
                f'section_channels must be positive widths, got {self.section_channels}'
            )


@dataclass(frozen=True)
class PpoConfig:
    total_env_steps: int
    rollout_steps: int
    epochs_per_update: int
    minibatch_size: int
    learning_rate: float
    adam_epsilon: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    checkpoint_interval_updates: int
    metrics_window_episodes: int

    @classmethod
    def from_config(cls, section: Mapping[str, object]) -> Self:
        config = cls(
            total_env_steps=require_int(section, 'total_env_steps'),
            rollout_steps=require_int(section, 'rollout_steps'),
            epochs_per_update=require_int(section, 'epochs_per_update'),
            minibatch_size=require_int(section, 'minibatch_size'),
            learning_rate=require_float(section, 'learning_rate'),
            adam_epsilon=require_float(section, 'adam_epsilon'),
            gamma=require_float(section, 'gamma'),
            gae_lambda=require_float(section, 'gae_lambda'),
            clip_epsilon=require_float(section, 'clip_epsilon'),
            value_loss_coef=require_float(section, 'value_loss_coef'),
            entropy_coef=require_float(section, 'entropy_coef'),
            max_grad_norm=require_float(section, 'max_grad_norm'),
            checkpoint_interval_updates=require_int(
                section, 'checkpoint_interval_updates'
            ),
            metrics_window_episodes=require_int(section, 'metrics_window_episodes'),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, count in (
            ('total_env_steps', self.total_env_steps),
            ('rollout_steps', self.rollout_steps),
            ('epochs_per_update', self.epochs_per_update),
            ('minibatch_size', self.minibatch_size),
            ('checkpoint_interval_updates', self.checkpoint_interval_updates),
            ('metrics_window_episodes', self.metrics_window_episodes),
        ):
            if count < 1:
                raise ValueError(f'{name} must be at least 1, got {count}')
        for name, positive in (
            ('learning_rate', self.learning_rate),
            ('adam_epsilon', self.adam_epsilon),
            ('clip_epsilon', self.clip_epsilon),
            ('max_grad_norm', self.max_grad_norm),
        ):
            if positive <= 0:
                raise ValueError(f'{name} must be positive, got {positive}')
        for name, fraction in (
            ('gamma', self.gamma),
            ('gae_lambda', self.gae_lambda),
        ):
            if not 0 <= fraction <= 1:
                raise ValueError(f'{name} must be in [0, 1], got {fraction}')
        for name, coef in (
            ('value_loss_coef', self.value_loss_coef),
            ('entropy_coef', self.entropy_coef),
        ):
            if coef < 0:
                raise ValueError(f'{name} must not be negative, got {coef}')


@dataclass(frozen=True)
class EvalConfig:
    seed: int
    episodes: int
    interval_updates: int

    @classmethod
    def from_config(cls, section: Mapping[str, object]) -> Self:
        config = cls(
            seed=require_int(section, 'seed'),
            episodes=require_int(section, 'episodes'),
            interval_updates=require_int(section, 'interval_updates'),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError(f'seed must not be negative, got {self.seed}')
        for name, count in (
            ('episodes', self.episodes),
            ('interval_updates', self.interval_updates),
        ):
            if count < 1:
                raise ValueError(f'{name} must be at least 1, got {count}')
