# Runtime

Originally resolved [#16](https://github.com/jenujari/prdict-pov-v1/issues/16) with a podman container. **That is history now** — the host moved from Chimera Linux (musl) to Arch Linux (glibc), where the whole ML stack installs natively. The container is deleted; this note records the native setup and why the container existed.

**Everything runs natively via `uv` on the host.**

```sh
uv sync                                 # install everything (once)
uv run python scripts/verify_env.py     # smoke test
uv run python scripts/build_column_spec.py
```

## Why there is no longer a container

The blocker #16 recorded was **musl libc**: `uv` installed a musl CPython, and neither `torch` nor `xgboost` publishes musl wheels, so the model half of the pipeline could not run on the host at all. The workaround was a glibc podman container that owned the entire runtime.

On **Arch Linux (glibc x86_64)** that constraint is gone — every wheel below resolves and installs directly:

| Package | Version |
|---|---|
| `pandas` / `numpy` / `scipy` / `scikit-learn` | 3.0.5 / 2.5.1 / 1.18.0 / 1.9.0 |
| `pyarrow` / `matplotlib` | 25.0.0 / 3.11.1 |
| `xgboost` | 3.3.0 |
| `torch` | 2.13.0 (CPU) |
| `pytorch-forecasting` | 1.8.0 |

These are the versions the container proved out (#5, #6); the floors in `pyproject.toml` track them.

## CPU-only torch

The box has **no GPU** (map #2 standing preferences). The default PyPI `torch` wheel drags in the full CUDA stack (~2 GB, useless here), so `pyproject.toml` pins torch to the PyTorch CPU index:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

`explicit = true` means only packages mapped to it in `[tool.uv.sources]` (i.e. `torch`) draw from that index; everything else stays on PyPI. Verify with `torch.version.cuda is None` in `scripts/verify_env.py`.

## Dependency manifest

There is now **one** manifest: `pyproject.toml` + `uv.lock`. (Under the container there were two — a host `pyproject.toml` for tooling and a `container/requirements.txt` for the runtime. That split is gone.) Add a dependency with `uv add`, re-lock with `uv lock`.

## Smoke test

`scripts/verify_env.py` imports the stack and *exercises* the two packages the musl host could not run:

- `torch` — a real matmul on a 64×8 tensor, and asserts `torch.version.cuda is None`.
- `xgboost` — `multi_strategy="multi_output_tree"` with `tree_method="hist"`, trained and predicting a 10-wide output (the exact combination #6 needs and map decision 2 rides on).

`pytorch-forecasting` 1.8.0 is the version the [#5 research](https://github.com/jenujari/prdict-pov-v1/issues/5) recommended.

## Note on `exchange_calendars`

The forward trading-day calendar (#4) was generated once with `exchange_calendars`; its output is frozen and checked into `kb/trading_calendar.json`. It is a one-off generator, **not** a pipeline dependency, so it is deliberately absent from `pyproject.toml`.
