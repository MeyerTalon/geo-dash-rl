# AGENTS.md

Guidance for AI coding agents working in this repo.

## Overview

A reinforcement learning agent that plays Geometry Dash from pixels. It trains online in a Rust simulator written for this project, in parallel and with any modification we like, and runs on the real game only at inference. The target is levels it has never seen. It runs on a Mac (Apple silicon first, then CUDA, then CPU). The design is in `docs/system-architecture.md`.

The work splits in two:

- Rust (`rust/`) is the simulator (`rust/engine/`), the PyO3 bindings that expose it to Python (`rust/bindings/`), and the level importer (`rust/importer/`). Real-game screen capture and input will also live here.
- Python (`python/src/`) holds the learning code and a thin typed layer over the simulator.

The repo was rewritten from scratch. Nothing from the earlier live-game version remains.

`mise.toml` at the root pins the toolchain (Python 3.11, uv, ripgrep, Rust 1.94) and defines every task. Go through mise rather than calling uv, python, or cargo directly: tasks run from the repo root, where every relative path resolves.

## Paved-path rules

- **Copy the existing pattern.** Before writing anything, find the closest existing example (module, function, test, config, task) and match it. If no pattern exists, propose one before inventing it.
- **No comments.** Not in code, configs, stubs, or `Cargo.toml`. Code explains itself through names, types, and structure.
- **Names and types are the documentation.** Units go in names (`timeout_s`, `FRAME_WIDTH_PX`). Docstrings are for what the signature can't show: a reason, an invariant, a precondition, an ordering, a safety constraint, or a tensor shape.
- **No magic numbers.** A literal whose meaning isn't stated where it's used gets a named constant. Hyperparameters are config keys, not constants.
- **Group digits with underscores.** Integers of four or more digits in code and configs use `_` separators: `1_000_000`, `2_000`, `65_535`. YAML, Python, and Rust all parse this form. Text in strings and prose is exempt.
- **Hyperparameters come from configs.** Every knob is a required key in `python/src/configs/`. No defaults in code.
- **No workarounds.** No `# noqa`, `# type: ignore`, `cast`, `Any`, `#[allow]`, skipped tests, broad `except Exception`, or special cases that paper over a wrong design. Untyped third-party APIs get a stub in `python/typings/`.
- **Never loosen enforcement.** Don't relax ruff, mypy, pytest, clippy, or the golden-path scan to make code pass. Tightening is welcome. Loosening needs explicit approval.
- **`mise run check` is the gate.** It passes on the whole repo today and must still pass before a change is done.

## Skills

Skills live in `.agents/skills/<name>/SKILL.md`. `.claude/skills`, `.codex/skills`, and `.cursor/skills` are symlinks to that tree (checked by `lint:golden-path`), so never create a real copy elsewhere. To add one, copy `_template`.

- `ponytail`: the laziest solution that works. Applies to every coding task.
- `python-coding`: rules for every `.py` and `.pyi` file.
- `ml-coding`: network, trainer, observation, config, and checkpoint rules. Composes with `python-coding`.
- `unslop`: cut AI tells and padding from prose, and the brevity mode ("/concise", "TL;DR"). Applies to docs, skills, docstrings, commit messages, and replies.

When the user corrects you, fix the instance and propose the layer that prevents it next time: the code, then a lint or test, then a skill, then this file.

## Commands

```bash
mise run setup # uv sync, then build the Rust engine into python/.venv
mise run rust:develop # rebuild the engine into python/.venv after a change under rust/
mise run rust:build # cargo build for every crate
mise run levels:import <source> <output> # convert a game level string file into a level json, for example levels/1.json
mise run run:bench # random-policy simulator throughput
mise run run:train --run-name <name> # train a policy
mise run run:eval --checkpoint <path> # evaluate a checkpoint on held-out levels
mise run run:tensorboard # live graphs for every run in runs/
mise run fmt # ruff format and cargo fmt
mise run check # lint + test: run before finishing any change
mise tasks # list all tasks
```

`check` runs `lint` and `test`:

- `lint`: `ruff format --check`, `ruff check`, `mypy` (strict, stubs from `python/typings/`), `cargo fmt --check`, `cargo clippy -D warnings` over the workspace, and `lint:golden-path`.
- `lint:golden-path`: fails on any comment in `.py`, `.pyi`, `.yml`, and `.rs` files or in `mise.toml`, `python/pyproject.toml`, and the Rust manifests, on integers of four or more digits without underscores in those code and config files, and on any of `.claude`, `.codex`, or `.cursor` whose `skills` isn't a symlink to `.agents/skills`.
- `test`: rebuilds the engine, then runs `pytest` over every `<pkg>/tests/` and `cargo test` for the workspace. Tests use tiny tensors and fakes: no game, no display, no network, a few seconds total.

Add Python dependencies with `uv add` (or `uv add --dev`) from `python/` and Rust dependencies with `cargo add` from the crate directory. Never hand-edit `uv.lock` or `Cargo.lock`. `uv sync` removes the built `geo_dash_native` module, so run `mise run rust:develop` after it (`setup` does).

Rust has no comments, and the golden-path scan forbids `//` and so doc comments too. Put the contract in names, types, `NonZero` integers, and the architecture doc. Public Rust functions don't return `Result`, because clippy's `missing_errors_doc` would demand a doc comment. Validate at the boundary (the bindings or the Python config) and keep the engine types valid by construction. The importer is a binary crate, so its private functions return `Result<_, String>`.

## Feature map

```
mise.toml                    toolchain and tasks (the CLI)
docs/system-architecture.md  the design, the interface contract, and the roadmap
.agents/skills/              agent skills (symlinked from .claude/, .codex/, and .cursor/)
rust/
  Cargo.toml                 workspace manifest and the shared lint table
  clippy.toml                panic-family lints relaxed in tests only
  engine/                    the simulator crate, no Python dependency
    src/action.rs            Action enum and the u8 conversion
    src/level.rs             Level, Rect, Spike, and seeded level generation
    src/level_pool.rs        LevelPool: imported levels mixed with generated ones per episode
    src/physics.rs           Player and the fixed-tick cube physics
    src/render.rs            ObsSpec and the grayscale frame renderer: player, ground, spikes, solids
    src/frame_stack.rs       pushing and refilling a stack of frames
    src/env.rs               EnvConfig, one Env, reward and termination
    src/vec_env.rs           parallel VecEnv with auto reset
  bindings/                  PyO3 crate built into the geo_dash_native module
    pyproject.toml           maturin build config
  importer/                  binary crate: game level string to level json
    src/decode.rs            base64url and gunzip of a level string
    src/parse.rs             level text to raw objects
    src/table.rs             object id to solid, spike, decoration, or not simulated
    src/convert.rs           raw objects to a Level, pixel to unit conversion, import report
    src/main.rs              command line entry point
python/
  pyproject.toml             deps, ruff, mypy, pytest
  uv.lock                    locked Python dependencies
  typings/                   stubs for the native module
  src/config.py              YAML loading, section references, and the require_* readers
  src/configs/default.yml    run manifest: seed, sim, model, ppo, and eval sections
  src/configs/impala_x2.yml  the same manifest with the wider model
  src/configs/imported.yml   the same manifest as default.yml with levels/1.json mixed into training
  src/configs/sim/           simulator settings: default.yml is generated levels only, imported.yml adds levels/1.json
  src/configs/model/         network dimensions
  src/configs/ppo/           PPO hyperparameters
  src/sim/config.py          SimConfig dataclass
  src/sim/vec_env.py         VecEnv protocol, StepBatch, NativeVecEnv
  src/model/config.py        ModelConfig, PpoConfig, and EvalConfig dataclasses
  src/model/network.py       IMPALA CNN and the ActorCritic policy
  src/model/advantage.py     GAE
  src/model/ppo.py           rollout batch, losses, and the PPO update
  src/model/checkpoint.py    atomic checkpoint save and load
  src/model/device.py        MPS, CUDA, or CPU selection
  src/rollout.py             rollout collection and held-out evaluation
  src/run_log.py             per-run CSV and TensorBoard metrics
  src/train.py               training entry point
  src/eval.py                checkpoint evaluation entry point
  src/bench_sim.py           random-policy throughput entry point
```

## Safety rules

- **Don't run real-game tasks yourself.** Any future task that captures the screen or sends input takes over the real mouse and keyboard and needs the game window open. Ask the user to run it. Simulator tasks like `run:bench` are safe.
- **The real-game player will assume a 1280x720 windowed game at the top-left of the primary display.** Nothing implements it yet.
- **Keep checkpoints loadable.** Once the network exists, module attribute names, `nn.Sequential` order, and checkpoint keys are a compatibility surface for saved weights. Changing them needs explicit user approval. Weights saved by the pre-rewrite network are not compatible with anything here.
- **Never commit weights or game levels.** `checkpoints/`, `runs/`, and `levels/` stay out of git. Imported levels are copyrighted game data.
- **Leave changes local by default.** Don't `git add`, commit, push, or open pull requests unless the user explicitly asks for that action.
- Never add yourself as a contributor, and never mention yourself in a commit message or pull request. No `Co-authored-by` trailers for an agent, model, or tool.
- Don't expand scope beyond the asked change.

## Conventions

- Follow `python-coding` and `ml-coding` for every `.py` file. `ponytail` applies to every coding task, and `unslop` to every piece of prose. The skills hold the detailed rules; don't duplicate them here.
- Strings use single quotes (enforced by `ruff format`).
- Docs can lag code. When they disagree, trust the code and fix the doc in the same change. The interface contract table in `docs/system-architecture.md` changes together with the engine, bindings, stub, wrapper, and tests.
