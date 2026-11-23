import argparse
from collections.abc import Sequence
from pathlib import Path

from src.config import require_section
from src.model.checkpoint import load_checkpoint
from src.model.config import EvalConfig, ModelConfig
from src.model.device import select_device
from src.model.network import ActorCritic
from src.rollout import eval_sim_config, evaluate_policy
from src.sim.config import SimConfig
from src.sim.vec_env import NativeVecEnv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = load_checkpoint(args.checkpoint)
    sim_config = SimConfig.from_config(require_section(checkpoint.config, 'sim'))
    model_config = ModelConfig.from_config(require_section(checkpoint.config, 'model'))
    eval_config = EvalConfig.from_config(require_section(checkpoint.config, 'eval'))

    device = select_device()
    model = ActorCritic(model_config)
    model.load_state_dict(checkpoint.model_state)
    result = evaluate_policy(
        NativeVecEnv(eval_sim_config(sim_config, eval_config), eval_config.seed),
        model.to(device),
        eval_config.episodes // sim_config.num_envs,
        device,
    )
    print(
        f'{result.episodes} episodes, '
        f'mean progress {result.mean_progress:.1%}, '
        f'completion rate {result.completion_rate:.1%}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
