---
name: python-coding
description: >
  Rules every piece of Python in this repo must follow: no comments or
  suppressions, names and types as the documentation (units in names,
  docstrings only for what the signature can't show), no magic numbers, no
  inline imports, small single-job modules, reuse before writing new code,
  deleting dead code, strict types with no Any, single quotes, and fast
  offline tests that never need the game or a screen. Use this skill whenever
  writing, editing, refactoring, or reviewing any Python in this repo (python/src/,
  tests, python/typings/ stubs), even for a small fix, and even if the user
  doesn't mention style or conventions.
---

# Python coding rules

These rules apply to every `.py` and `.pyi` file. The goal is code that the next
module imports and builds on instead of copy-pasting.

Repo context: Python 3.11, `uv` for deps, ruff for format and lint (single
quotes), mypy `--strict` with stubs in `python/typings/`, pytest, all run through
`mise`. `mise run check` enforces type hints, no `Any`, import order,
keyword-only booleans, and no comments. The rules below cover what the tools
can't judge.

Simulation, rendering, and real-game screen capture and input belong in
`rust/`. Python holds learning code and a thin typed layer over the Rust
simulator. Don't reimplement simulator logic in Python.

## 0. No comments, no suppressions

- No `#` comments anywhere, including `# noqa`, `# type: ignore`, and
  `# pragma`. If code needs one, rename or restructure it.
- No `cast`, no `Any`, and no config changes that loosen ruff or mypy. Fix the
  code.
- An untyped library call gets a typed local (`frame: np.ndarray = ...`) or a
  stub in `python/typings/`, declaring only what the code uses (`geo_dash_native`, for
  example).
- A workaround for a platform quirk (the macOS OpenMP duplicate-library error)
  belongs in the environment or the mise task, not in an `os.environ` line at
  the top of a module.

## 1. Layers and modules

- Packages are layered, lowest first:

  | Package | Owns | May import |
  |---|---|---|
  | `python/src/config.py` | reading typed values from YAML | nothing in the project |
  | `python/src/sim/` | typed access to the Rust simulator | `src.config` |
  | `python/src/model/` | the network, the trainer, and their configs | `src.config` |
  | entry points (`python/src/bench_sim.py`, `python/src/train.py`, `python/src/eval.py`) and their shared wiring (`python/src/rollout.py`, `python/src/run_log.py`) | wiring the packages together | everything |

  `sim` and `model` never import each other. The entry point turns observation
  arrays into tensor batches and calls the trainer. If a lower module
  needs something from a higher one, the thing is in the wrong place: move it
  down.
- One job per module, named for that job. Past about 300 lines, or when
  describing a module needs an "and", split it along its jobs into sibling
  modules. A module's tests split with it.
- No `utils.py`, `helpers.py`, or `common.py`. A helper belongs to a named job
  or starts one.
- Entry points are thin: parse args, load the config, call library functions,
  print or save, return an exit code. Logic a second script could want belongs
  in a package.
- Separate computing from presenting. A function that computes a number doesn't
  print, plot, or log it. Return data; callers decide what to do with it.
- Read config once at the edge and pass typed values down. Library functions
  take paths, devices, and settings as parameters, never globals or env vars.
- Functions do one thing: don't mix stepping, parsing, and file I/O in one body.
- Imports go at the top of the module, always. Platform-only imports
  (`AppKit`) are top-level too, behind a stub in `python/typings/`, not inside a
  method.
- Every module is in the Feature map in `AGENTS.md` with one line on what it
  owns.

## 2. Reuse before you write

Each new script starts as a copy of the nearest one, so a duplicate spreads and
two copies of one rule drift apart. Before writing a function, type, or
constant, search `python/src/` for one that already does the job and import it within
the layer rules above.

- **Extend before you add.** When a function almost fits, give it a parameter
  or split out the shared part. Never copy it to change one line.
- **Extract on the second copy**, into the lowest module both callers share,
  and switch both callers in the same change.
- **A new module needs a reason.** Add one only when no existing module owns
  the job, and tell the user which ones you ruled out.
- No base classes or plugin systems for a single implementation.

## 3. Types

- Use built-in generics and unions: `dict[str, int]`, `list[Action]`,
  `Path | None`. Never `Dict`, `List`, `Optional`, or `Tuple`.
- Config sections are frozen dataclasses with a `from_config` classmethod that
  reads every key without a fallback and validates ranges.
- Closed sets are `StrEnum`s or `Enum`s, not `str` or `int`. A `match` over
  one ends with `case _: assert_never(value)`.
- Fixed-shape records are `NamedTuple`s or frozen dataclasses, not tuples or
  `dict[str, int]` (an observation size is a pair of named fields, not
  `{'w': ..., 'h': ...}`).
- Structural interfaces are `Protocol`s, so tests can pass fakes (a frame
  source, an input sink).
- Boolean parameters are keyword-only.
- Bad input types raise `TypeError`; bad values raise `ValueError`. No `assert`
  for validation, no bare `except`, no broad `except Exception`.

## 4. Names and types are the documentation

- A name says what a thing is and carries its unit: `timeout_s`,
  `FRAME_WIDTH_PX`, `step_delay_s`, `clip_epsilon`, `progress_fraction`.
- A test name states the behavior it checks
  (`test_done_flag_stops_advantage_from_crossing_episodes`).
- Docstrings are the exception, not the default. Write one only for what the
  signature can't show, in one or two plain sentences:
  - a reason;
  - an invariant;
  - a precondition (actions hold exactly one entry per env);
  - an ordering;
  - a safety constraint (this takes over the real mouse, in the future play code);
  - a tensor shape (`(B, T)` or `(B, C, H, W)`).
- Never restate the name or signature, and no `Args:` / `Returns:` sections.

## 5. No magic numbers

- A literal whose meaning isn't stated where it's used becomes a module-level
  constant whose name carries the meaning and unit: observation size, key
  delays, frame cadence, loss coefficients, kernel sizes.
- Derive related constants from each other (`FRAME_PERIOD_S = 1 / TARGET_FPS`).
- Never reach into a structure by position to mean something
  (`shape[:2]`, `row[3]`, `Path(__file__).parents[2]`): name it or look it up.
- Hyperparameters are not module constants. They are required config keys (see
  `ml-coding`).
- A literal is fine where a name at the same spot already says what it is (a
  keyword argument, a config key, a dataclass field), as `0`/`1` identities, as
  format widths, and as test fixtures and expected values.
- Ruff `PLR2004` enforces this for comparisons (relaxed in tests only).
  Everything else is on you.

## 6. Tests

- Tests live in `<pkg>/tests/test_<module>.py` and run in seconds: tiny
  tensors, `tmp_path` for every written file, CPU, no network, no GPU.
- Tests never need Geometry Dash, a display, or input permissions. Fake the
  environment and input boundaries with `Protocol` implementations or
  `monkeypatch`. The built Rust extension is allowed.
- Every non-trivial branch in preprocessing, advantage estimation, loss
  masking, or checkpoint logic gets a test, using small inputs whose answer you
  can compute by hand.

## 7. Delete dead code

Dead code is deleted in the change that makes it dead. Git keeps the history;
the repo keeps only what runs. Delete:

- functions, classes, constants, and modules nothing calls or reads;
- empty modules and placeholder files (a `trainer.py` with no content);
- the old path once its replacement lands, never both "just in case";
- parameters every caller passes the same value, and flags nothing sets;
- tests that only exercise deleted code, and Feature map and doc lines that
  describe it.

Never comment code out or hide it behind a leading `_`. ruff and mypy don't
flag an unused public function, so before finishing, grep for every name your
change stopped calling and delete what is left without a caller.

## Before finishing

Run `mise run fmt` and `mise run check`. Then re-read the diff: nothing
duplicated that could have been imported, no dead code left behind, no module
grown past about 300 lines, units in names, no magic numbers, no docstring that
restates its item, new modules and tasks added to the Feature map and Commands
in `AGENTS.md`.
