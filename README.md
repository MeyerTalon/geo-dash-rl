# geo-dash-rl

A reinforcement learning agent that plays Geometry Dash from pixels. It trains in a simulator written for this project and runs on the real game only at inference. The target is levels the agent has never seen. It runs on macOS.

The full design is in [docs/system-architecture.md](docs/system-architecture.md).

## How it works

1. A Rust engine reimplements the game's physics and renders frames. It steps many environments in parallel.
2. A PyTorch learner trains an IMPALA-style CNN policy with PPO-style updates against that engine.
3. A real-game player (not built yet) captures the screen, feeds the same preprocessing to the trained policy, and presses the mouse.

This is online RL in a simulator, not offline RL and not a world model.

## Status

The repository is mid-rewrite. The old version trained in the live game with OCR on the level percentage, and none of it remains. What exists now:

- A Rust engine with placeholder cube physics, generated spike levels, a grayscale renderer, and a parallel vectorized environment.
- PyO3 bindings and a typed Python wrapper (`python/src/sim/`).
- An IMPALA CNN with a PPO trainer, plus train and eval entry points.
- A level importer that converts the game's official level files into engine levels (`rust/importer/`).
- A random-policy benchmark.

The real-game player is planned. See the roadmap in the architecture doc.

## Layout

- `rust/engine/` is the simulator: physics, levels, renderer, vectorized environment.
- `rust/bindings/` builds the engine into the Python module `geo_dash_native`.
- `rust/importer/` converts game level files into engine levels. Output goes to `levels/`, which stays out of git.
- `python/src/sim/` is the typed Python layer over the engine.
- `python/src/config.py` and `python/src/configs/` read YAML configs. Every hyperparameter is a required key.
- `python/src/bench_sim.py` steps the simulator with random actions and reports throughput.
- `python/typings/` holds type stubs for the native module.
- `docs/` holds the architecture doc.
- `AGENTS.md` and `.agents/skills/` hold the rules for AI coding agents.

## Setup

[mise](https://mise.jdx.dev) pins Python 3.11, uv, ripgrep, and Rust 1.94, and defines every task.

```bash
brew install mise
mise install
mise run setup
```

`mise run setup` installs the Python dependencies into `python/.venv` and builds the Rust engine into it. Run `mise run rust:develop` after any change under `rust/` to rebuild the module.

Training needs no game install, no Tesseract, and no screen permissions.

## Commands

Run everything from the repository root.

```bash
mise run setup          # uv sync, then build the engine into python/.venv
mise run rust:develop   # rebuild the engine into python/.venv
mise run rust:build     # cargo build for every crate
mise run run:bench      # random-policy simulator throughput
mise run fmt            # ruff format and cargo fmt
mise run check          # lint and test
mise tasks              # list all tasks
```
