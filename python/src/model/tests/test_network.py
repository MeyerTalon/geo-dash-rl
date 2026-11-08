import torch

from src.model.config import ModelConfig
from src.model.network import ActorCritic

REAL_CONFIG = ModelConfig(
    input_channels=4,
    input_height_px=72,
    input_width_px=128,
    section_channels=(16, 32, 32),
    residual_blocks_per_section=2,
    feature_dim=256,
    action_count=2,
)
TINY_CONFIG = ModelConfig(
    input_channels=4,
    input_height_px=18,
    input_width_px=32,
    section_channels=(4, 8),
    residual_blocks_per_section=1,
    feature_dim=16,
    action_count=2,
)
ENCODER_PARAMETER_COUNT = 1_277_648


def random_frames(config: ModelConfig, batch_size: int) -> torch.Tensor:
    shape = (
        batch_size,
        config.input_channels,
        config.input_height_px,
        config.input_width_px,
    )
    return torch.randint(0, 256, shape, dtype=torch.uint8)


def test_encoder_matches_the_documented_parameter_count() -> None:
    model = ActorCritic(REAL_CONFIG)
    encoder_parameters = [*model.sections.parameters(), *model.features.parameters()]
    assert sum(p.numel() for p in encoder_parameters) == ENCODER_PARAMETER_COUNT


def test_forward_returns_logits_and_values_per_sample() -> None:
    logits, values = ActorCritic(TINY_CONFIG)(random_frames(TINY_CONFIG, 3))
    assert logits.shape == (3, TINY_CONFIG.action_count)
    assert values.shape == (3,)


def test_forward_and_backward_run_at_the_real_input_shape() -> None:
    model = ActorCritic(REAL_CONFIG)
    logits, values = model(random_frames(REAL_CONFIG, 16))
    (logits.sum() + values.sum()).backward()
    assert next(model.sections.parameters()).grad is not None


def test_samples_in_a_batch_do_not_affect_each_other() -> None:
    model = ActorCritic(TINY_CONFIG)
    frames = random_frames(TINY_CONFIG, 3)
    batched_logits, batched_values = model(frames)
    single_logits, single_values = model(frames[1:2])
    assert torch.allclose(batched_logits[1:2], single_logits, atol=1e-5)
    assert torch.allclose(batched_values[1:2], single_values, atol=1e-5)


def test_deterministic_action_is_the_highest_logit() -> None:
    model = ActorCritic(TINY_CONFIG)
    frames = random_frames(TINY_CONFIG, 4)
    logits, _ = model(frames)
    step = model.get_action(frames, deterministic=True)
    assert torch.equal(step.actions, logits.argmax(dim=-1))


def test_evaluate_actions_matches_the_sampled_log_probs() -> None:
    model = ActorCritic(TINY_CONFIG)
    frames = random_frames(TINY_CONFIG, 4)
    step = model.get_action(frames, deterministic=False)
    evaluation = model.evaluate_actions(frames, step.actions)
    assert torch.allclose(evaluation.log_probs, step.log_probs)
