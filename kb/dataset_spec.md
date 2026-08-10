# Dataframe schema and the two model layouts

Resolves [#11](https://github.com/jenujari/prdict-pov-v1/issues/11). Defined and read by `prdict/dataset.py`; nothing is generated ahead of time — the schema is a pure function of the encoding spec ([#9](https://github.com/jenujari/prdict-pov-v1/issues/9)), the calendar ([#4](https://github.com/jenujari/prdict-pov-v1/issues/4)) and the target ([#8](https://github.com/jenujari/prdict-pov-v1/issues/8)).

**Scope.** Physical layout only. What each family *becomes* is `kb/encoding_spec.md`; this is where those encoded columns are arranged into the one long frame the TFT reads and the flat matrix XGBoost reads. Decided at the 2026-08-08 realignment — see [`docs/adr/0001`](../docs/adr/0001-input-window-is-60-past-plus-30-future.md) (window is 60 past + 30 future = 90) and [`docs/adr/0002`](../docs/adr/0002-tft-decodes-30-scores-10.md) (TFT decodes 30, scores 10).

## The long-format frame (item 1)

One row per **history** trading session, `n = 6517` rows. `TimeSeriesDataSet` *is* the windower — it slices the encoder-60 / decoder-30 windows itself in `__getitem__` — so there is no hand-rolled tensor; there is only this frame plus the covariate lists below.

| Column | dtype | Role |
|--------|-------|------|
| `time_idx` | `int32` | Dense rank of the trading days, `0 … 6516`. |
| `series_id` | `category` | Constant `"nifty50"` — one instrument, one series. |
| `y` | `float32` | Target: the one-session log return landing on the session, `y[i] = log(C_i / C_{i-1})`. |
| `elapsed` | `float32` | Known-future covariate: calendar days the step landing on the session spans. |
| 280 features | per `kb/encoding_spec.md` | The encoded astronomical block, `tft` view. |

- **`time_idx` is integer and dense by construction.** It ranks the trading-day index, not the calendar, so consecutive sessions differ by exactly 1 with no holes. `allow_missing_timesteps=False` therefore holds and nothing needs filling — the self-check asserts `time_idx.diff() == 1` throughout. Nothing else assumes calendar spacing: every span that *is* calendar-sensitive is carried explicitly by `elapsed`.
- **`y[0] = 0` and `elapsed[0] = 0`** are a convention, not data. The first session has no predecessor, so its return and step are undefined; the row is kept only so `time_idx` stays equal to `cal.position`. The first trainable origin is the 60th session, so this fabricated pair is never a *target* — it rides only in the target-history channel of the earliest encoder windows.
- **`nft50.csv` is newest-first.** The frame is built through `trading_calendar.restrict()`, which sorts ascending, so row `i` is `cal.sessions[i]`. Reading the CSV in file order would silently negate every return; nothing here does.

## The four covariate lists (item 2)

Populated from each family's `tft_role` in the encoding spec, plus `elapsed` by hand.

| List | n | Members |
|------|---|---------|
| `time_varying_known_reals` | **214** | `boolean` (14) + `cyclic` (92) + `linear_numeric` (107) + `elapsed` (1) |
| `time_varying_known_categoricals` | **67** | `categorical` |
| `time_varying_unknown_reals` | **0** | — |
| `time_varying_unknown_categoricals` | **0** | — |

**Per [#3](https://github.com/jenujari/prdict-pov-v1/issues/3) every feature is known-future** — the ephemeris is computable for the decoder block exactly as for the encoder — so the unknown lists are empty. The only quantity the model may not see past the origin is the return it is predicting, and `TimeSeriesDataSet` carries that through `target="y"`, never through an unknown list. This is stated, not implied.

`elapsed` is a **decoder-side known-future covariate (D4)**: one real here, `elapsed_1..10` in the XGBoost matrix. It is **not** a member of the 280-column block and must not trip #9's width assertion — the self-check asserts `elapsed not in spec.all_columns`.

**Scaling is not delegated to the library.** `linear_numeric` is fold-scaled in `prdict.encoding.build_fold` before the frame reaches `make_dataset`, so every real's internal `TimeSeriesDataSet` scaler is disabled (`scalers={col: None}`) — otherwise the library would refit a second, dataset-wide scaler that sees validation rows. The categorical encoders passed in are the pre-fitted **global** `NaNLabelEncoder`s, so no fold sizes an embedding table from what it happened to observe (map decision 6).

## Origin alignment (item 3)

For an origin at `time_idx = t`:

| Block | Sessions | Meaning |
|-------|----------|---------|
| Encoder | `t-59 … t` | 60 sessions up to and including the origin |
| Decoder | `t+1 … t+30` | 30 known-future sessions the TFT decodes |
| Scored | `t+1 … t+10` | the first 10 decoder steps, the only ones compared |
| Target | `y[t+k] = log(C_{t+k}/C_{t+k-1})`, `k = 1…10` | map decision 2 |

Verified two ways in `main()`: `y[t+k]` is checked cell-for-cell against the raw closes at a probe origin, **and** one sample is pulled from a constructed `TimeSeriesDataSet` and its decoder target confirmed to equal `frame.y[t+1 … t+10]`. So the library's own windowing agrees with the hand arithmetic — the encoder/decoder split is where this schema says it is.

## XGBoost — the full-flatten matrix (items 5–6)

XGBoost has no windower, so `dataset.flatten()` builds it. **One flat row per origin**, the 90 window sessions × 280 features, then `elapsed_1..10`:

```
90 sessions × 280 features + 10 elapsed = 25,210 columns
```

- **Column order** is offset-major, feature-minor: `col@t-59 … col@t+30` for every feature, then `elapsed_1..elapsed_10`. This is exactly the C-order flatten of a `(90, 280)` window, so the layout is derivable, not stored. Names and `feature_types` are fold-invariant, built once by `flatten_columns`.
- **Categoricals ride as integer codes** with a matching `feature_types` entry of `"c"` (67 × 90 = 6030 columns), so a `DMatrix(enable_categorical=True)` treats them as categorical rather than ordinal. The numeric families and `elapsed` are `"q"`. The single float32 array avoids a 25,210-column mixed-dtype DataFrame.
- **The window reaches into the forward block.** A late scored origin's decoder (`t+30`) sits past the last observed close; the features there are real known-future values (#3), so `flatten` builds its feature matrix over **all** 6663 sessions, not just history. Only `y` is history-bounded.
- **Built per fold, measured.** One fold's train block is **2470 origins × 25,210 × 4 B ≈ 249 MB**; the whole trainable history would be **≈ 650 MB**. That is why the matrix is a per-fold call, not a materialised artefact. The flatten happens **after** encoding and **before** [#12](https://github.com/jenujari/prdict-pov-v1/issues/12)'s stage-1 `|r| ≥ 0.95` prune, which collapses the near-collinear day-lags these columns hold. **#11 and #12 are a coupled pair.**

## The `folds.py` ↔ `encoding.py` wiring (C3)

Neither module reached for the other; this ticket connects them:

- `dataset.build_fold_views(spec, frame, fold, cal)` is the one call the training loop makes per fold. It names the rows the scaler may see with `fold_fit_rows` and hands them to `encoding.build_fold`, then returns the fitted `FoldState` (which refuses to pickle) and the scaled frame.
- `fold_fit_rows(fold, cal)` returns the sessions `train_start-59 … train_end` — every session the fold's training origins read through their **encoder**. Decoder-future sessions are excluded: they cross the 30-session purge toward the validation block, and although that encoder overlap is harmless (every feature is known-future, PR #34), the scaler fit stops at the origin so the boundary needs no argument.
- `encoding.assert_fit_boundary(spec)` runs **once** at pipeline start (asserted in `main()`), checking the global/fold split structurally.

## Reading the schema

```python
from prdict.dataset import tft_frame, make_dataset, covariates, flatten, build_fold_views
from prdict.trading_calendar import load_calendar
from prdict import encoding

cal = load_calendar()
spec = encoding.load_spec()
g = encoding.load_global(spec)

frame = tft_frame(cal, spec, g)          # (6517, 284) long frame
cov = covariates(spec)                    # the four lists
ds = make_dataset(frame, cal, spec, g)    # TimeSeriesDataSet (TFT)

# per fold:
state, scaled = build_fold_views(spec, frame, fold, cal)   # C3 wiring
fm = flatten(cal, spec, g, fold.train)    # (n, 25210) XGBoost matrix
```

```sh
uv run python -m prdict.dataset    # self-check: alignment, wiring, memory
```
