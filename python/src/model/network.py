from itertools import pairwise
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.config import ModelConfig

CONV_KERNEL_PX = 3
CONV_PADDING_PX = 1
POOL_KERNEL_PX = 3
POOL_STRIDE_PX = 2
POOL_PADDING_PX = 1
PIXEL_MAX = 255.0


class PolicyStep(NamedTuple):
    """Every field is `(B,)`."""

    actions: Tensor
    log_probs: Tensor
    values: Tensor


class ActionEvaluation(NamedTuple):
    """Every field is `(B,)`."""

    log_probs: Tensor
    entropy: Tensor
    values: Tensor


def action_log_probs(log_policy: Tensor, actions: Tensor) -> Tensor:
    """`log_policy` is `(B, A)` and `actions` is `(B,)`. Returns `(B,)`."""
    return log_policy.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


class ResidualBlock(nn.Module):
    """Hand-rolled because torch has no pre-activation residual block. The ReLUs must stay
    out of place, because `x` also feeds the skip path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(
            channels, channels, CONV_KERNEL_PX, padding=CONV_PADDING_PX
        )
        self.conv_b = nn.Conv2d(
            channels, channels, CONV_KERNEL_PX, padding=CONV_PADDING_PX
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden: Tensor = self.conv_a(F.relu(x))
        correction: Tensor = self.conv_b(F.relu(hidden))
        return x + correction


class ConvSection(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, residual_blocks: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, CONV_KERNEL_PX, padding=CONV_PADDING_PX
        )
        self.pool = nn.MaxPool2d(
            POOL_KERNEL_PX, stride=POOL_STRIDE_PX, padding=POOL_PADDING_PX
        )
        self.blocks = nn.Sequential(
            *(ResidualBlock(out_channels) for _ in range(residual_blocks))
        )

    def forward(self, x: Tensor) -> Tensor:
        downsampled: Tensor = self.pool(self.conv(x))
        output: Tensor = self.blocks(downsampled)
        return output


class ActorCritic(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = (config.input_channels, *config.section_channels)
        self.sections = nn.Sequential(
            *(
                ConvSection(
                    in_channels, out_channels, config.residual_blocks_per_section
                )
                for in_channels, out_channels in pairwise(channels)
            )
        )
        blank_frames = torch.zeros(
            1, config.input_channels, config.input_height_px, config.input_width_px
        )
        flat_size = self.sections(blank_frames).numel()
        self.features = nn.Linear(flat_size, config.feature_dim)
        self.policy_head = nn.Linear(config.feature_dim, config.action_count)
        self.value_head = nn.Linear(config.feature_dim, 1)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """`obs` is `(B, C, H, W)` uint8. Returns logits `(B, A)` and values `(B,)`."""
        conv_output: Tensor = self.sections(obs.float() / PIXEL_MAX)
        features = F.relu(self.features(F.relu(conv_output).flatten(1)))
        logits: Tensor = self.policy_head(features)
        values: Tensor = self.value_head(features)
        return logits, values.squeeze(-1)

    def get_action(self, obs: Tensor, *, deterministic: bool) -> PolicyStep:
        logits, values = self(obs)
        log_policy = F.log_softmax(logits, dim=-1)
        actions = (
            logits.argmax(dim=-1)
            if deterministic
            else torch.multinomial(log_policy.exp(), 1).squeeze(-1)
        )
        return PolicyStep(actions, action_log_probs(log_policy, actions), values)

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> ActionEvaluation:
        logits, values = self(obs)
        log_policy = F.log_softmax(logits, dim=-1)
        entropy = -(log_policy.exp() * log_policy).sum(dim=-1)
        return ActionEvaluation(action_log_probs(log_policy, actions), entropy, values)
