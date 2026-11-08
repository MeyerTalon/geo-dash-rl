from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from src.config import require_float, require_int

U8_MAX = 255
U16_MAX = 65_535


@dataclass(frozen=True)
class SimConfig:
    num_envs: int
    frame_skip: int
    frame_stack: int
    obs_width_px: int
    obs_height_px: int
    level_length_units: float
    death_penalty: float
    completion_bonus: float

    @classmethod
    def from_config(cls, section: Mapping[str, object]) -> Self:
        config = cls(
            num_envs=require_int(section, 'num_envs'),
            frame_skip=require_int(section, 'frame_skip'),
            frame_stack=require_int(section, 'frame_stack'),
            obs_width_px=require_int(section, 'obs_width_px'),
            obs_height_px=require_int(section, 'obs_height_px'),
            level_length_units=require_float(section, 'level_length_units'),
            death_penalty=require_float(section, 'death_penalty'),
            completion_bonus=require_float(section, 'completion_bonus'),
        )
        config.validate()
        return config

    @property
    def obs_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_envs,
            self.frame_stack,
            self.obs_height_px,
            self.obs_width_px,
        )

    def validate(self) -> None:
        if self.num_envs < 1:
            raise ValueError(f'num_envs must be at least 1, got {self.num_envs}')
        for name, count in (
            ('frame_skip', self.frame_skip),
            ('frame_stack', self.frame_stack),
        ):
            if not 1 <= count <= U8_MAX:
                raise ValueError(f'{name} must be in [1, {U8_MAX}], got {count}')
        for name, size_px in (
            ('obs_width_px', self.obs_width_px),
            ('obs_height_px', self.obs_height_px),
        ):
            if not 1 <= size_px <= U16_MAX:
                raise ValueError(f'{name} must be in [1, {U16_MAX}], got {size_px}')
        if self.level_length_units <= 0:
            raise ValueError(
                f'level_length_units must be positive, got {self.level_length_units}'
            )
        if self.death_penalty < 0:
            raise ValueError(
                f'death_penalty must not be negative, got {self.death_penalty}'
            )
        if self.completion_bonus < 0:
            raise ValueError(
                f'completion_bonus must not be negative, got {self.completion_bonus}'
            )
