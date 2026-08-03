# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Python project managed with `uv`. Python version is pinned to 3.12 (see `.python-version`).

## Runtime — read this first

**The pipeline runs in a podman container, not on the host.** The host is Chimera Linux (musl libc) and cannot install `torch` or `xgboost` at all — no musl wheels exist. Full detail in `kb/runtime.md`.

```sh
./container/build.sh                              # build the image (once)
./container/run.sh python container/verify.py     # smoke test
./container/run.sh python scripts/<script>.py     # run anything
```

- Inside the container, call `python` directly. **Never `uv run`** — the working directory is the mounted repo, so it would reach for `/work/.venv`, which is the host's musl venv and will not execute.
- ML dependencies live in `container/requirements.in`, locked to `container/requirements.txt` by `./container/lock.sh`. **Do not add them to `pyproject.toml`** — they cannot resolve on musl.
- `--network=host` is required for podman here (tailscale DNS); `build.sh` and `run.sh` already pass it.

## Host environment

The host `uv` environment is for lightweight tooling only.

- Dependencies and the virtual environment are managed by `uv` — do not use `pip` directly.
- Install/sync dependencies: `uv sync`
- Run a command in the host environment: `uv run <command>`
- Add a dependency: `uv add <package>` (use `--dev` for dev-only dependencies)
- Regenerate the lockfile after editing dependencies: `uv lock`

## Conventions

- `pyproject.toml` is the source of truth for **host** dependencies; `uv.lock` is committed and should be kept in sync (run `uv lock` after any dependency change). `container/requirements.txt` is the source of truth for the **pipeline** runtime.
- Target Python 3.12 syntax and standard library features.

## Planning

Work on this project is planned as a wayfinder map at **[issue #2](https://github.com/jenujari/prdict-pov-v1/issues/2)**, with decision tickets as sub-issues. Read the map's "Settled while charting" table before proposing anything about windows, targets, encoding, or evaluation — those decisions are already made. Ticket blocking is a `Blocked by: #N` line at the bottom of each issue body.
