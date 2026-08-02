# prdict-pov-v1

A Python project managed with [uv](https://docs.astral.sh/uv/).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

Clone the repository and sync dependencies:

```bash
git clone <repo-url>
cd prdict-pov-v1
uv sync
```

`uv sync` creates a `.venv` and installs dependencies from `uv.lock`.

## Usage

Run the project:

```bash
uv run main.py
```

## Development

Add a dependency:

```bash
uv add <package>
```

Add a dev-only dependency:

```bash
uv add --dev <package>
```

Run a command inside the project's environment:

```bash
uv run <command>
```

Update the lockfile after changing dependencies:

```bash
uv lock
```
