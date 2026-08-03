# prdict-pov-v1

Testing whether Vedic-astrology ephemeris features predict forward Nifty-50 index returns.

Daily NSE Nifty-50 closes are joined to planetary longitudes, speeds, nakshatras, signs,
vedhas, balas, and cross-planet angular separations. Three models are trained on the same
target — an always-predict-up baseline, XGBoost, and a Temporal Fusion Transformer — and
compared on a trading-simulation scorecard. No price-derived features are used, so the
comparison tests the astro hypothesis rather than price momentum.

## Running anything

**The pipeline runs in a podman container, not on the host.** The host is Chimera Linux
(musl libc) and cannot install `torch` or `xgboost` at all — no musl wheels exist. Full
detail in [`kb/runtime.md`](kb/runtime.md).

```sh
./container/build.sh                              # build the image (once)
./container/run.sh python container/verify.py     # smoke test
./container/run.sh python scripts/<script>.py     # run anything
./container/run.sh                                # interactive shell
```

Inside the container, call `python` directly — never `uv run`.

## Dependencies

Two manifests, deliberately:

| | Manifest | For |
|---|---|---|
| Pipeline | `container/requirements.in` → `container/requirements.txt` | The runtime that executes everything. Relock with `./container/lock.sh`. |
| Host | `pyproject.toml` + `uv.lock` | Lightweight tooling only. `uv sync`, `uv add <package>`. |

ML dependencies go in the first and never in the second — they cannot resolve on musl.

## Layout

| Path | |
|---|---|
| `nft50.csv` | The dataset. 9893 rows × 240 columns, 2000-01-01 → 2027-01-31. Given, never regenerated. |
| `categories_list.json` | Full declared level list for every categorical column, including levels never observed. |
| `container/` | Image definition, lockfile, build/run/lock scripts, smoke test. |
| `kb/` | Written specs and research notes. |
| `prdict/` | Library code. |
| `scripts/` | Runnable scripts. |
| `backup.one.txt` | An earlier notebook attempt. Reference only — several of its choices are explicitly overridden. |

## Planning

Work is planned as a wayfinder map at
[issue #2](https://github.com/jenujari/prdict-pov-v1/issues/2), with decision tickets as
sub-issues. Read the map's "Settled while charting" table before proposing anything about
windows, targets, encoding, or evaluation — those decisions are already made. Ticket
blocking is a `Blocked by: #N` line at the bottom of each issue body.
