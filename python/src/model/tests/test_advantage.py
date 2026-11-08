import torch

from src.model.advantage import compute_gae

GAMMA = 0.5
GAE_LAMBDA = 0.5
REWARDS = torch.tensor([[1.0, 1.0]])
VALUES = torch.tensor([[1.0, 1.0]])
LAST_VALUES = torch.tensor([2.0])


def advantages_with(dones: list[bool]) -> torch.Tensor:
    return compute_gae(
        REWARDS, VALUES, torch.tensor([dones]), LAST_VALUES, GAMMA, GAE_LAMBDA
    )


def test_advantage_bootstraps_from_the_last_value_without_a_done() -> None:
    assert torch.allclose(advantages_with([False, False]), torch.tensor([[0.75, 1.0]]))


def test_done_flag_stops_advantage_from_crossing_episodes() -> None:
    assert torch.allclose(advantages_with([True, False]), torch.tensor([[0.0, 1.0]]))


def test_terminal_last_step_does_not_bootstrap() -> None:
    assert torch.allclose(advantages_with([False, True]), torch.tensor([[0.5, 0.0]]))
