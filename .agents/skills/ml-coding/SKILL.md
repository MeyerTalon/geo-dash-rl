---
name: ml-coding
description: >-
  This repo's reinforcement-learning structure and practice: the simulator
  kept apart from learning, native PyTorch building blocks,
  documented tensor shapes, required-key configs as the only source of
  hyperparameters, one observation preprocessing path, tested GAE and PPO
  math, compatible checkpoints, strict-eval inference, and consumer-hardware
  defaults. Trigger whenever the user invokes "/ml-coding" or asks to add,
  edit, or review the network, trainer, rollouts, rewards, observations,
  configs, or checkpoints, even when the request is implicit. Composes with
  python-coding (style) and ponytail.
---

# ML coding

The agent learns to play Geometry Dash from rendered frames with an
IMPALA-style CNN and PPO-style actor-critic updates. It trains online in the
Rust simulator (see `docs/system-architecture.md`) on an Apple silicon laptop,
and only acts in the real game at inference. Python style comes from
`python-coding`; this skill covers learning structure and practice.

## Rules

- **Keep the simulator out of the learner.** `src/model/` never imports
  `src/sim/`. The trainer consumes batches of states, actions, rewards, dones,
  and old log-probs, so it runs on random tensors with no simulator built.
  Rollout collection lives in the entry point and talks to a `VecEnv`
  protocol.
- **Native PyTorch first.** Compose `nn.Conv2d`, `nn.LSTM`, `nn.Linear`,
  `nn.LayerNorm`, `torch.distributions`. Hand-roll a block (the IMPALA
  residual block) only when no built-in exists, and say why in its docstring.
- **Shape discipline.** Batch first everywhere. Time-major data is `(B, T)`
  for scalars and `(B, T, C, H, W)` for frames. `forward` returns raw value
  and logits and documents its shapes; sampling lives in `get_action`, never
  in `forward`.
- **Configs are the only source of hyperparameters.** `gamma`, `gae_lambda`,
  `clip_epsilon`, `value_loss_coef`, `entropy_coef`, `max_grad_norm`, learning
  rate, and network widths are required keys in `src/configs/*.yml`, read into
  a frozen dataclass. No default in code, no `config.get(key, fallback)`. A
  new knob goes into every config in the same change. Copy the pattern in
  `src/sim/config.py`: a frozen dataclass whose `from_config` reads each key
  with the `require_*` helpers in `src/config.py` and validates ranges.
- **One observation path.** The simulator renders frames and the real-game
  player captures them, and both go through one shared preprocessing step
  (resize, grayscale, scale, stacking) before the model. Its output shape and
  scaling are a compatibility surface, because checkpoints trained on one
  preprocessing break under another, and a mismatch between simulator and real
  frames is the main sim-to-real risk.
- **Test the math.** GAE, the clipped policy loss, value loss masking, and the
  done-flag handling get tests on tiny tensors with hand-computed answers: one
  episode boundary, one terminal bootstrap, a clipped ratio and an unclipped
  one.
- **Checkpoints are a compatibility surface.** Save the model state, optimizer
  state, config, and episode number together under `checkpoints/`. Don't
  rename module attributes, reorder `nn.Sequential`s, or change checkpoint keys
  without explicit user approval: existing weights stop loading.
- **Inference is strict eval.** Rebuild from the checkpoint's own config, call
  `load_state_dict`, `eval()`, and run under `torch.no_grad()`. Acting
  deterministically is a flag, not a different code path.
- **Seed once, at the entry point.** One `seed` config key seeds `random`,
  `numpy`, and `torch`. Library code never seeds.
- **Consumer hardware.** One function picks MPS, then CUDA, then CPU. Size the
  network and batch for the laptop's unified memory, and reach larger effective
  batches by accumulating gradients, not by growing a single batch.
- **Report data, not prints.** The trainer returns per-update losses, and the
  rollout returns per-episode return and progress fraction. The entry point
  decides where they go.
- **Verify by exercising.** After a meaningful change run `mise run check`,
  then one forward and backward pass on random CPU tensors of the real input
  shape. Simulator runs (`mise run run:bench`) are safe to start. A real-game
  play run needs the game window and takes over the mouse, so ask the user to
  run it and never start one yourself.

## Anti-goals

Don't introduce new RL or ML frameworks or tooling (gymnasium environments,
Stable-Baselines3, RLlib, TorchRL, Lightning, Hydra, wandb) unless asked. The
vectorized environment is the Rust `VecEnv` behind the `src/sim/` protocol,
not a gymnasium wrapper. Don't add a code-level default for a hyperparameter, skip checkpoint metadata, put sampling in `forward`, or
change the observation path without saying so. Don't touch the reward or the
done signal quietly: they define the task.
