from typing import NamedTuple, Protocol

import geo_dash_native
import numpy as np
from numpy.typing import NDArray

from src.sim.config import SimConfig

ACTION_COUNT = 2


class StepBatch(NamedTuple):
    """`obs` is `(N, S, H, W)` uint8 with the newest frame last along `S`. Envs that finished
    an episode are already reset, so every frame in their stack is the next episode's first
    frame while the other fields describe the finished one."""

    obs: NDArray[np.uint8]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    progress_fraction: NDArray[np.float32]


class VecEnv(Protocol):
    def reset(self) -> NDArray[np.uint8]: ...

    def step(self, actions: NDArray[np.uint8]) -> StepBatch: ...


class NativeVecEnv:
    def __init__(self, config: SimConfig, seed: int) -> None:
        self.obs_shape = config.obs_shape
        self.engine = geo_dash_native.VecEnv(
            config.num_envs,
            seed,
            geo_dash_native.EnvConfig(
                config.frame_skip,
                config.frame_stack,
                config.level_length_units,
                config.obs_width_px,
                config.obs_height_px,
                config.death_penalty,
                config.completion_bonus,
            ),
        )

    def reset(self) -> NDArray[np.uint8]:
        return self.decode_obs(self.engine.reset())

    def step(self, actions: NDArray[np.uint8]) -> StepBatch:
        obs, rewards, dones, progress_fraction = self.engine.step(actions.tobytes())
        return StepBatch(
            obs=self.decode_obs(obs),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            progress_fraction=np.asarray(progress_fraction, dtype=np.float32),
        )

    def decode_obs(self, obs: bytes) -> NDArray[np.uint8]:
        return np.frombuffer(obs, dtype=np.uint8).reshape(self.obs_shape).copy()
