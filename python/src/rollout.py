from dataclasses import replace
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from src.model.config import EvalConfig
from src.model.network import ActorCritic
from src.model.ppo import RolloutBatch
from src.sim.config import SimConfig
from src.sim.vec_env import VecEnv

COMPLETED_PROGRESS_FRACTION = 1.0


class EnvState(NamedTuple):
    """`obs` is `(N, C, H, W)` uint8, and `episode_returns` is `(N,)`."""

    obs: NDArray[np.uint8]
    episode_returns: NDArray[np.float32]


class Rollout(NamedTuple):
    batch: RolloutBatch
    next_state: EnvState
    finished_returns: list[float]
    finished_progress_fractions: list[float]


class EvalResult(NamedTuple):
    mean_progress: float
    completion_rate: float
    episodes: int


def initial_state(obs: NDArray[np.uint8]) -> EnvState:
    return EnvState(obs, np.zeros(obs.shape[0], dtype=np.float32))


def eval_sim_config(sim_config: SimConfig, eval_config: EvalConfig) -> SimConfig:
    return replace(sim_config, level_paths=eval_config.level_paths)


def completion_rate(progress_fractions: list[float]) -> float:
    return float(np.mean(np.array(progress_fractions) >= COMPLETED_PROGRESS_FRACTION))


@torch.no_grad()
def collect_rollout(
    env: VecEnv,
    model: ActorCritic,
    state: EnvState,
    rollout_steps: int,
    device: torch.device,
) -> Rollout:
    """Each step's `dones` entry says that step ended the episode, so the next stored
    observation already belongs to the next episode."""
    obs_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    log_prob_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    reward_steps: list[torch.Tensor] = []
    done_steps: list[torch.Tensor] = []
    finished_returns: list[float] = []
    finished_progress_fractions: list[float] = []
    obs = state.obs
    episode_returns = state.episode_returns.copy()

    for _ in range(rollout_steps):
        obs_tensor = torch.from_numpy(obs)
        policy_step = model.get_action(obs_tensor.to(device), deterministic=False)
        actions = policy_step.actions.cpu()
        result = env.step(actions.numpy().astype(np.uint8))

        obs_steps.append(obs_tensor)
        action_steps.append(actions)
        log_prob_steps.append(policy_step.log_probs.cpu())
        value_steps.append(policy_step.values.cpu())
        reward_steps.append(torch.from_numpy(result.rewards))
        done_steps.append(torch.from_numpy(result.dones))

        episode_returns += result.rewards
        finished_returns.extend(episode_returns[result.dones].tolist())
        finished_progress_fractions.extend(
            result.progress_fraction[result.dones].tolist()
        )
        episode_returns[result.dones] = 0.0
        obs = result.obs

    _, last_values = model(torch.from_numpy(obs).to(device))
    return Rollout(
        batch=RolloutBatch(
            obs=torch.stack(obs_steps, dim=1),
            actions=torch.stack(action_steps, dim=1),
            log_probs=torch.stack(log_prob_steps, dim=1),
            values=torch.stack(value_steps, dim=1),
            rewards=torch.stack(reward_steps, dim=1),
            dones=torch.stack(done_steps, dim=1),
            last_values=last_values.cpu(),
        ),
        next_state=EnvState(obs, episode_returns),
        finished_returns=finished_returns,
        finished_progress_fractions=finished_progress_fractions,
    )


@torch.no_grad()
def evaluate_policy(
    env: VecEnv, model: ActorCritic, episodes_per_env: int, device: torch.device
) -> EvalResult:
    """Every env plays the same number of episodes, because keeping the first episodes to
    finish would favor early deaths. Terminates because levels have a finite length."""
    model.eval()
    obs = env.reset()
    finished_counts = np.zeros(obs.shape[0], dtype=np.int64)
    progress_fractions: list[float] = []
    while (finished_counts < episodes_per_env).any():
        policy_step = model.get_action(
            torch.from_numpy(obs).to(device), deterministic=True
        )
        result = env.step(policy_step.actions.cpu().numpy().astype(np.uint8))
        counted = result.dones & (finished_counts < episodes_per_env)
        progress_fractions.extend(result.progress_fraction[counted].tolist())
        finished_counts += result.dones
        obs = result.obs
    model.train()
    return EvalResult(
        mean_progress=float(np.mean(progress_fractions)),
        completion_rate=completion_rate(progress_fractions),
        episodes=len(progress_fractions),
    )
