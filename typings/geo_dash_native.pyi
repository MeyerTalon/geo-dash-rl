class EnvConfig:
    def __init__(
        self,
        frame_skip: int,
        level_length_units: float,
        obs_width_px: int,
        obs_height_px: int,
        death_penalty: float,
        completion_bonus: float,
    ) -> None: ...

class VecEnv:
    def __init__(self, num_envs: int, seed: int, config: EnvConfig) -> None: ...
    def reset(self) -> bytes: ...
    def step(
        self, actions: bytes
    ) -> tuple[bytes, list[float], list[bool], list[float]]: ...
