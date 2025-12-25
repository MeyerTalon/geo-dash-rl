"""Trainer class for IMPALA-style CNN reinforcement learning."""

from __future__ import annotations

# Workaround for OpenMP duplicate library issue on macOS
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ImpalaCNN


class ImpalaTrainer:
    """Trainer for IMPALA-style CNN using PPO-style actor-critic loss.

    Args:
        model: The ImpalaCNN model to train.
        dataloader: DataLoader providing batches of (states, actions, rewards, dones, ...).
        optimizer: Optimizer for model parameters.
        device: Device to run training on ('cpu' or 'cuda').
        gamma: Discount factor for future rewards.
        gae_lambda: GAE lambda parameter for advantage estimation.
        clip_epsilon: PPO clip epsilon for policy loss.
        value_loss_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
        log_dir: Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        model: ImpalaCNN,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        log_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor of shape (B, T) or (B,).
            values: Value estimates of shape (B, T) or (B,).
            dones: Done flags of shape (B, T) or (B,).
            next_values: Next state values if available, else None.

        Returns:
            Tuple of (advantages, returns).
        """
        # Ensure all tensors are 2D: (B, T)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if values.dim() == 1:
            values = values.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)

        # Ensure all have same shape
        batch_size, seq_len = rewards.shape
        assert values.shape == (batch_size, seq_len), f"Values shape {values.shape} != rewards shape {rewards.shape}"
        assert dones.shape == (batch_size, seq_len), f"Dones shape {dones.shape} != rewards shape {rewards.shape}"

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        if next_values is None:
            next_values = torch.zeros(batch_size, 1, device=rewards.device)
        elif next_values.dim() == 1:
            next_values = next_values.unsqueeze(1)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = next_values
            else:
                next_value = values[:, t + 1 : t + 2]

            delta = rewards[:, t : t + 1] + self.gamma * next_value * (1 - dones[:, t : t + 1]) - values[:, t : t + 1]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t : t + 1]) * last_gae
            advantages[:, t : t + 1] = last_gae

        returns = advantages + values
        return advantages.squeeze(-1), returns.squeeze(-1)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute PPO-style actor-critic loss.

        Args:
            states: Input states of shape (B, C, H, W) or (B, T, C, H, W).
            actions: Actions taken of shape (B,) or (B, T).
            old_log_probs: Old log probabilities of shape (B,) or (B, T).
            advantages: Advantage estimates of shape (B,) or (B, T).
            returns: Return estimates of shape (B,) or (B, T).
            hidden: LSTM hidden state if using LSTM.

        Returns:
            Dictionary with loss components.
        """
        # Forward pass
        values, logits, _ = self.model(states, hidden)
        values = values.squeeze(-1)

        # Policy loss (PPO clip)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    def train_step(
        self,
        batch: dict[str, Any] | tuple[Any, ...],
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Batch from dataloader. Expected to contain:
                - states: (B, C, H, W) or (B, T, C, H, W)
                - actions: (B,) or (B, T)
                - rewards: (B, T) or (B,)
                - dones: (B, T) or (B,)
                - old_log_probs: (B,) or (B, T)
                - hidden: Optional LSTM hidden state

        Returns:
            Dictionary of loss metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Handle different batch formats
        if isinstance(batch, dict):
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            dones = batch["dones"].to(self.device)
            old_log_probs = batch.get("old_log_probs", torch.zeros_like(actions)).to(self.device)
            hidden = batch.get("hidden", None)
            if hidden is not None:
                hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
        else:
            # Assume tuple format: (states, actions, rewards, dones, ...)
            states = batch[0].to(self.device)
            actions = batch[1].to(self.device)
            rewards = batch[2].to(self.device)
            dones = batch[3].to(self.device)
            old_log_probs = batch[4].to(self.device) if len(batch) > 4 else torch.zeros_like(actions).to(self.device)
            hidden = batch[5] if len(batch) > 5 else None
            if hidden is not None:
                hidden = (hidden[0].to(self.device), hidden[1].to(self.device))

        # Get value estimates for GAE
        with torch.no_grad():
            values, _, _ = self.model(states, hidden)
            values = values.squeeze(-1)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages (skip if only 1 element to avoid std() error)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # For single element, just center it
            advantages = advantages - advantages.mean()

        # Compute loss
        loss_dict = self.compute_loss(states, actions, old_log_probs, advantages, returns, hidden)

        # Backward pass
        loss_dict["total_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Optimizer step
        self.optimizer.step()

        # Convert to float for logging
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        metrics["grad_norm"] = self._get_grad_norm()

        self.step += 1
        return metrics

    def _get_grad_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1.0 / 2)

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of average metrics over the epoch.
        """
        all_metrics = []
        for batch in self.dataloader:
            metrics = self.train_step(batch)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        self.epoch += 1
        return avg_metrics

    def save_checkpoint(self, filename: str | None = None) -> Path:
        """Save model checkpoint.

        Args:
            filename: Optional filename. If None, uses step number.

        Returns:
            Path to saved checkpoint.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}_step_{self.step}.pt"

        checkpoint_path = self.log_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)

    def train(
        self,
        num_epochs: int,
        save_every: int | None = None,
        log_every: int = 10,
    ) -> None:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
            save_every: Save checkpoint every N epochs. If None, only save at end.
            log_every: Print metrics every N steps.
        """
        for epoch in range(num_epochs):
            metrics = self.train_epoch()

            if epoch % log_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Total Loss: {metrics['total_loss']:.4f}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Grad Norm: {metrics['grad_norm']:.4f}")
                print()

            if save_every is not None and epoch % save_every == 0:
                self.save_checkpoint()

        # Save final checkpoint
        self.save_checkpoint("final_checkpoint.pt")


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset

    # Create dummy model and data
    model = ImpalaCNN(
        input_shape=(3, 720, 1280), 
        num_actions=3, 
        use_lstm=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # # Dummy dataset
    # states = torch.randn(1, 3, 720, 1280)
    # actions = torch.randint(0, 3, (1,))
    # rewards = torch.randn(1, 1)
    # dones = torch.zeros(1, 1)
    # old_log_probs = torch.randn(1)

    # dataset = TensorDataset(states, actions, rewards, dones, old_log_probs)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # # Create trainer
    # trainer = ImpalaTrainer(
    #     model=model,
    #     dataloader=dataloader,
    #     optimizer=optimizer,
    #     device="cpu",
    #     log_dir="src/model/checkpoints",
    # )

    # # Train
    # trainer.train(num_epochs=1, log_every=1)
