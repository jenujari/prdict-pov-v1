# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Python project managed with `uv`. Python version is pinned to 3.12 (see `.python-version`).

## Runtime — read this first

**The whole pipeline runs natively via `uv` on the host.** The host is Arch Linux (glibc), so `torch`, `xgboost`, and `pytorch-forecasting` install directly — no container. Full detail in `kb/runtime.md`. (This project used to require a podman container to dodge musl; that host is gone, and so is `container/`.)

```sh
uv sync                                    # install everything (once)
uv run python scripts/verify_env.py        # smoke test
uv run python scripts/<script>.py          # run anything
```

- `torch` is CPU-only by construction — it resolves from the PyTorch CPU index declared in `pyproject.toml` (`[tool.uv.sources]` + `[[tool.uv.index]]`). This box has **no CUDA**; size the TFT accordingly.
- ML dependencies live in `pyproject.toml` like everything else. Add one with `uv add <package>`; re-lock with `uv lock`.

## Host environment

- Dependencies and the virtual environment are managed by `uv` — do not use `pip` directly.
- Install/sync dependencies: `uv sync`
- Run a command: `uv run <command>`
- Add a dependency: `uv add <package>` (use `--dev` for dev-only dependencies)
- Regenerate the lockfile after editing dependencies: `uv lock`

## Conventions

- `pyproject.toml` is the single source of truth for dependencies; `uv.lock` is committed and kept in sync (run `uv lock` after any dependency change).
- Target Python 3.12 syntax and standard library features.

## Planning

Work on this project is planned as a wayfinder map at **[issue #2](https://github.com/jenujari/prdict-pov-v1/issues/2)**, with decision tickets as sub-issues. Read the map's "Settled while charting" table before proposing anything about windows, targets, encoding, or evaluation — those decisions are already made. Ticket blocking is a `Blocked by: #N` line at the bottom of each issue body.
