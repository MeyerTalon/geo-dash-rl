"""IMPALA-style CNN for 2D platformer reinforcement learning."""

from __future__ import annotations

# Workaround for OpenMP duplicate library issue on macOS
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from game_engine.action_engine import Action

class ImpalaCNN(nn.Module):
    """IMPALA-style CNN for processing 2D platformer game frames.

    Architecture:
        - Feature extractor: 3 convolutional blocks
        - LSTM: Optional temporal processing
        - Value head: Estimates state value
        - Policy head: Outputs action probabilities

    Args:
        input_shape: (C, H, W) of input frames. Default assumes RGB 1280x720.
        num_actions: Number of discrete actions (e.g., 2 for jump/no-jump).
        use_lstm: If True, add LSTM layer for temporal dependencies.
        lstm_size: Hidden size of LSTM if enabled.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (3, 720, 1280),
        num_actions: int = 2,
        use_lstm: bool = False,
        lstm_size: int = 256,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.lstm_size = lstm_size

        # Feature extractor: 3 conv blocks (IMPALA style)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Compute flattened size after convs
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            self.conv_output_size = dummy.numel()

        # Optional LSTM for temporal processing
        if use_lstm:
            self.lstm = nn.LSTM(self.conv_output_size, lstm_size, batch_first=True)
            feature_size = lstm_size
        else:
            self.lstm = None
            feature_size = self.conv_output_size

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through the network.

        Args:
            x: Input frames of shape (B, C, H, W) or (B, T, C, H, W) if using LSTM.
            hidden: LSTM hidden state (h, c) if using LSTM, else None.

        Returns:
            Tuple of:
                - value: State value estimates (B, 1) or (B, T, 1) if LSTM.
                - logits: Action logits (B, num_actions) or (B, T, num_actions) if LSTM.
                - hidden: Updated LSTM hidden state, or None if not using LSTM.
        """
        # Handle temporal dimension
        if self.use_lstm and x.dim() == 5:
            batch_size, seq_len = x.shape[:2]
            x = x.view(batch_size * seq_len, *x.shape[2:])
            temporal = True
        else:
            temporal = False

        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # LSTM processing
        if self.use_lstm:
            if temporal:
                x = x.view(batch_size, seq_len, -1)
            else:
                x = x.unsqueeze(1)  # Add temporal dimension

            x, hidden = self.lstm(x, hidden)
            if not temporal:
                x = x.squeeze(1)

        # Value and policy heads
        value = self.value_head(x)
        logits = self.policy_head(x)

        return value, logits, hidden

    def get_action(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Sample an action from the policy.

        Args:
            x: Input frame(s) of shape (B, C, H, W) or (B, T, C, H, W).
            hidden: LSTM hidden state if using LSTM.
            deterministic: If True, take the most likely action; else sample.

        Returns:
            Tuple of:
                - action: Sampled action indices (B,) or (B, T,).
                - log_prob: Log probability of the action.
                - value: State value estimate.
                - hidden: Updated LSTM hidden state.
        """
        value, logits, hidden = self.forward(x, hidden)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1), hidden


if __name__ == "__main__":

    # Example usage
    model = ImpalaCNN(
        input_shape=(3, 720, 1280), 
        num_actions=3, 
        use_lstm=True
    )

    # Single frame
    frame = torch.randn(1, 3, 720, 1280)
    value, logits, hidden = model(frame)
    print(f"Single frame - Value shape: {value.shape}, Logits shape: {logits.shape}")

    # Action sampling
    action, log_prob, value, hidden = model.get_action(frame)
    action = Action(action.item())
    print(f"Action: {action.name}, Log prob: {log_prob}, Value: {value}")

    # Temporal sequence
    sequence = torch.randn(1, 4, 3, 720, 1280)  # (B, T, C, H, W)
    value, logits, hidden = model(sequence)
    print(f"Sequence - Value shape: {value.shape}, Logits shape: {logits.shape}")
