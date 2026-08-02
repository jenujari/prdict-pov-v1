# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Python project managed with `uv`. Python version is pinned to 3.12 (see `.python-version`).

## Environment & commands

- Dependencies and the virtual environment are managed by `uv` — do not use `pip` directly.
- Install/sync dependencies: `uv sync`
- Run the project: `uv run main.py`
- Run any command in the project environment: `uv run <command>`
- Add a dependency: `uv add <package>` (use `--dev` for dev-only dependencies)
- Regenerate the lockfile after editing dependencies: `uv lock`

## Conventions

- `pyproject.toml` is the source of truth for dependencies; `uv.lock` is committed and should be kept in sync (run `uv lock` after any dependency change).
- Target Python 3.12 syntax and standard library features.
