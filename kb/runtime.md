# Runtime

Resolves [#16](https://github.com/jenujari/prdict-pov-v1/issues/16).

**Everything in this project runs inside a podman container.** The host cannot install the ML stack.

```sh
./container/build.sh                              # build the image (once)
./container/run.sh python container/verify.py     # smoke test
./container/run.sh python scripts/build_column_spec.py
./container/run.sh                                # interactive shell
```

## Why

The host is **Chimera Linux**, which uses **musl libc**. `uv` correctly installs a musl CPython, and most of the scientific-Python wheel ecosystem does not publish musl builds.

| Package | On the musl host | In the container |
|---|---|---|
| `pandas`, `numpy`, `scipy`, `scikit-learn` | Installs | 3.0.5 / 2.5.1 / 1.18.0 / 1.9.0 |
| `xgboost` | **No musl wheel.** `uv` backtracks to 2.0.3, which then needs `cmake`. | **3.3.0** |
| `torch` | **No musl wheel at any version.** Wheels exist only for `manylinux_2_28_{x86_64,aarch64}`, `macosx_14_0_arm64`, `win_amd64`. | **2.13.0+cpu** |
| `pytorch-forecasting` | Unreachable — needs torch | 1.8.0 |

The XGBoost gap is a packaging accident and is separately solvable — the [#6 research](https://github.com/jenujari/prdict-pov-v1/issues/6) got 3.3.0 to build from source on musl in about 2.5 minutes. The PyTorch gap is not: building it on musl is unsupported upstream and a multi-hour toolchain exercise.

Rather than run a split environment, **the container is the single runtime for the whole pipeline** — feature engineering included, even though `pandas` and `scikit-learn` would run natively. One environment means one lockfile and one set of library versions, so a result cannot depend on where it was run. Verified: `scripts/build_column_spec.py` produces byte-identical output on the host and in the container.

The host `uv` environment is retained only for lightweight tooling. Do not add model dependencies to `pyproject.toml` — they cannot resolve on musl.

## Layout

| Path | Purpose |
|---|---|
| `container/Containerfile` | Image definition. Base `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` (glibc 2.36). |
| `container/requirements.in` | Direct dependencies, hand-edited. |
| `container/requirements.txt` | Fully pinned lock — 47 packages. Generated, do not hand-edit. |
| `container/lock.sh` | Regenerates the lock. Run after editing `requirements.in`. |
| `container/build.sh` | Builds the image. |
| `container/run.sh` | Runs a command in the container with the repo mounted at `/work`. |
| `container/verify.py` | Smoke test — imports everything, exercises torch and xgboost, checks the mount is writable. |

Image is ~3.95 GB.

## Reproducibility

`container/requirements.txt` is generated on the **musl host** by cross-resolving for glibc:

```sh
uv pip compile container/requirements.in \
    --python-platform x86_64-manylinux_2_28 \
    --python-version 3.12 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -o container/requirements.txt
```

`--python-platform` is what makes this work: it resolves for the target platform rather than the host, so torch and xgboost resolve on a machine that cannot install either. Wrapped in `container/lock.sh`.

The `download.pytorch.org/whl/cpu` index pins `torch==2.13.0+cpu`. Without it, pip resolves the default wheel and drags in the full CUDA stack — pointless on a box with no GPU, and roughly 2 GB of it.

This project therefore has **two** dependency manifests. `pyproject.toml` + `uv.lock` describe the host tooling environment; `container/requirements.txt` describes the runtime that actually executes the pipeline. The second is the one that matters for results.

## Gotchas

**`--network=host` was required, and no longer is.** Podman on this machine used to inherit the host's tailscale resolver (`nameserver 100.100.100.100`) into a network namespace that could not reach it, so the base-image pull failed with `lookup ghcr.io on 100.100.100.100:53: no such host` and both scripts passed `--network=host` to work around it. The host DNS configuration was fixed on 2026-08-03 — containers now get `8.8.8.8` / `1.1.1.1` in their own namespace. Verified: DNS, an HTTPS fetch to PyPI, and a `podman build` that installs from PyPI all succeed with no network flag.

The flag has been dropped from `build.sh` and `run.sh`, so the container runs in its own network namespace again. If a pull or a build ever fails on DNS, check the host's resolver before reaching for `--network=host`.

**Never call `uv run` inside the container.** The working directory is the mounted repo, so `uv run` would try to sync against `pyproject.toml` and either fight the image's pinned environment or reach for `/work/.venv` — which is the host's **musl** venv and will not execute. `UV_PROJECT_ENVIRONMENT=/opt/venv` is set in the image as a guard, but the rule is simply: call `python` directly. It resolves to `/opt/venv/bin/python` via `PATH`.

**File ownership is already correct.** Rootless podman maps the container's root to the invoking host user, so files written to `/work` come back owned by that user. No `--userns=keep-id` needed. Verified.

**CPU only.** `torch.version.cuda` is `None` by construction. Eight threads available.

## Wired for the models

Both blocked models were exercised, not merely imported — see `container/verify.py`:

- `torch` — a real matmul on a 64x8 tensor.
- `xgboost` — `multi_strategy="multi_output_tree"` with `tree_method="hist"`, trained and predicting a 10-wide output. This is the exact combination [#6](https://github.com/jenujari/prdict-pov-v1/issues/6) needed and the map's decision 2 rides on.

`pytorch-forecasting` 1.8.0 is the version the [#5 research](https://github.com/jenujari/prdict-pov-v1/issues/5) recommended pinning to.
