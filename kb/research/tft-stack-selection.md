# TFT implementation stack selection (CPU-only)

Date researched: 2026-08-02

Which Temporal Fusion Transformer implementation should this project use to predict a 10-step vector of
forward Nifty-50 log returns from ~200 mostly-known-future Vedic-astrology ephemeris features, on a
CPU-only box, with ~6500 single-series windows and ~40 declared-cardinality categoricals?

---

## Recommendation

**Use `pytorch-forecasting` >= 1.8.0 (`TimeSeriesDataSet` + `TemporalFusionTransformer`, the v1 API), on
CPU torch.** Let the library own windowing. Do **not** hand-roll a past-60/future-30 tensor pipeline.

Why:

- **It is the only candidate whose covariate model matches this project 1:1.** `time_varying_known_reals`,
  `time_varying_known_categoricals`, `time_varying_unknown_reals`, `time_varying_unknown_categoricals`,
  `static_*` are literal constructor arguments
  ([`_timeseries.py` L451-456](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L451-L456)).
  Darts has no time-varying categorical embeddings at all, and NeuralForecast's TFT explicitly
  refuses categorical exogenous variables (see Q5).
- **Declared cardinality can be *forced*, including levels never observed.** Pre-fit a
  `NaNLabelEncoder(add_nan=True)` on the full level list from `categories_list.json` and pass it in
  `categorical_encoders={col: enc}`. `TimeSeriesDataSet._preprocess_data` calls `check_is_fitted` and
  **will not re-fit an already-fitted encoder**
  ([L1129-1138](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L1129-L1138)),
  and `TemporalFusionTransformer.from_dataset` derives `embedding_sizes` as
  `(len(encoder.classes_), get_embedding_size(len(encoder.classes_)))`
  ([`_base_model.py` L2002-2007](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/base/_base_model.py#L2002-L2007)).
  Unobserved declared levels get their own embedding row; `add_nan=True` reserves index 0 for
  genuinely unknown values so inference never raises.
- **Irregular trading-day index is a non-issue**: `time_idx` must be an *integer* column
  (`assert data[self.time_idx].dtype.kind == "i"`,
  [L1000-1003](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L1000-L1003)).
  Rank-encode trading days 0..N-1 and there are no gaps to fill and no `freq=` to declare.
- **Direct 10-step multi-horizon with `QuantileLoss` by default** — faithful to the TFT paper
  ([arXiv:1912.09363](https://arxiv.org/abs/1912.09363)); nothing is recursive.
- **Best maintained of the three on the axes that matter here**: 7 releases in the last 18 months,
  113 commits in the last 52 weeks, latest commit 2026-08-01, CI on Python 3.10–3.14 (Q1 table).
- **There is still a full escape hatch** if the dataframe route ever fails: every hyperparameter
  (`x_reals`, `x_categoricals`, `embedding_sizes`, `time_varying_reals_encoder`, …) is a plain
  constructor argument
  ([`_tft.py` L139-172](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py#L139-L172)),
  and the batch dict contract is explicit
  ([L2525-2540](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L2525-L2540)),
  so a custom `Dataset`/`DataLoader` can be swapped in without forking the model.

Honest trade-offs / what you give up:

- **The messiest issue tracker of the three**: 627 open issues, 162 open PRs (vs Darts 192/23,
  NeuralForecast 4/4). Much of this is the in-flight "v2" API rewrite. Mitigation: pin `==1.8.x` and
  stay on the **v1** API — `_timeseries_v2.py` and `_tft_v2.py` both carry explicit
  "experimental / not for production" disclaimers.
- **You give up the pre-built-tensor design.** `TimeSeriesDataSet` builds windows from a long
  dataframe; the FE pipeline must emit rows, not windows. This is the constraint the sample-tensor
  ticket must absorb (see the FE section below).
- **You give up Darts' ergonomics** (backtesting helpers, `historical_forecasts`) and
  NeuralForecast's speed-tuned dataloader. The walk-forward purge/embargo CV harness has to be
  written by hand around `TimeSeriesDataSet.from_dataset(..., min_prediction_idx=...)`.
- **The 200-variable Variable Selection Network is a Python `for` loop over variables**
  ([`sub_modules.py` L363-382](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.py#L363-L382)),
  so CPU cost is dispatch-bound, not FLOP-bound. Manageable at this scale but it is the thing that
  will hurt (Q7).

**Runner-up worth knowing about:** GluonTS `torch` TFT has, on paper, the *cleanest* API for this exact
problem — `dynamic_cardinalities` / `past_dynamic_cardinalities` are declared cardinalities passed
directly to the estimator, no encoder-fitting round trip. It loses on maintenance: 23 commits in 52
weeks, 342 open issues, 127 open PRs, and a 12-month release gap (0.16.2 → 0.16.3). Keep it as the
fallback if pytorch-forecasting's categorical plumbing disappoints in the prototype.

---

## Comparison table

| Axis | pytorch-forecasting 1.8.0 | Darts 0.46.1 | NeuralForecast 3.2.0 | GluonTS 0.17.0 (torch) |
|---|---|---|---|---|
| 1. Maintained 2026 | Yes. 7 rel/18mo, 113 commits/52wk, last commit 2026-08-01. **627 open issues / 162 PRs.** Py 3.10–3.14 in CI | Yes. 18 rel/18mo, 185 commits/52wk, last commit 2026-07-31. 192 issues / 23 PRs. CI 3.10 + 3.12 | Yes. 14 rel/18mo, 143 commits/52wk, last 2026-07-28. 4 issues / 4 PRs. CI 3.10–3.13 | Marginal. 5 rel/18mo but a 12-mo gap; 23 commits/52wk. 342 issues / 127 PRs. Py <3.15,>=3.10 |
| 2. Covariate split | **First class**: `time_varying_known_reals`, `time_varying_unknown_reals`, `time_varying_known_categoricals`, `time_varying_unknown_categoricals`, `static_reals`, `static_categoricals` | First class for reals: `past_covariates`, `future_covariates`, `static_covariates`. `future_covariates` mandatory | First class: `futr_exog_list`, `hist_exog_list`, `stat_exog_list` | First class: `dynamic_dims`, `past_dynamic_dims`, `static_dims` + `*_cardinalities` |
| 3. Windowing owner | Library. Input = long `pd.DataFrame`. Escape hatch: construct model from explicit hparams + custom `Dataset` emitting the documented batch dict | Library. Input = `TimeSeries` objects. Escape hatch: **`fit_from_dataset(TorchTrainingDataset)`** (public, documented) | Library. Input = long df, or `TimeSeriesDataset` passed to `model.fit(dataset=...)`. Windows always cut inside `_create_windows` | Library. Input = iterable of dicts (`start`, `target`, `feat_dynamic_real`, `feat_dynamic_cat`, …); `TFTInstanceSplitter` cuts windows |
| 4. Irregular index | **No freq at all.** `time_idx` must be integer. Set `allow_missing_timesteps=False` and rank-encode trading days → dense, nothing to fill | Needs `pd.RangeIndex` (integer step) or a `DatetimeIndex` with inferable freq. Trading-day `DatetimeIndex` fails; `fill_missing_dates=True, freq="B"` would inject NaN holiday rows | `freq: Union[str, int]`; an integer `ds` column with `freq=1` is explicitly supported by `validate_freq` | `start` is a `pd.Period`; use a synthetic daily index + `time_features=[]` |
| 5. Categoricals | **Native embeddings, forceable cardinality** via pre-fitted `NaNLabelEncoder` + `add_nan=True`; unobserved declared levels get their own row | **Static covariates only.** `categorical_embedding_sizes` feeds `static_covariates_vsn`. Time-varying categoricals must be one-hot/encoded upstream as reals | `cat_exog_list` + `categorical_cardinalities` exist — **but `TFT` does not set `EXOGENOUS_CAT = True`, so it raises**. Also vocab is built from *observed* uniques; unobserved levels collapse to OOV index 0 | `dynamic_cardinalities=[...]` declared directly; embedding table sized from the declaration, no data inspection |
| 6. Multi-horizon | Direct. `max_prediction_length=10`, `QuantileLoss` default; MAE/RMSE/MAPE/SMAPE/MASE/Tweedie/Poisson + distributional; no recursion | Direct within `output_chunk_length`; autoregressive **only** if `n > output_chunk_length`. `QuantileRegression` default | Direct (`RECURRENT = False`). `MAE` default; `MQLoss`, `QuantileLoss`, `HuberMQLoss`, `DistributionLoss`, `sCRPS`, … | Direct. `QuantileOutput` default |
| 7. CPU viability | ~0.28 TFLOP/epoch at `hidden_size=16` (set 1); dispatch-bound. ~10–60 s/epoch order of magnitude | Similar arithmetic; `skip_interpolation=True` speed flag exists | **More expensive per variable**: every exog gets a full `hidden×hidden` GRN (PTF reals use `hidden_continuous_size=8` and cats use a near-free `ResampleNorm`) | Comparable to PTF |
| Verdict | **CHOSEN** | Rejected: no time-varying categorical embeddings | Rejected: TFT refuses categorical exog | Fallback |

---

## 1. Maintenance status as of 2026

All figures fetched **2026-08-02** via `gh api` (GitHub REST) and the PyPI JSON API.

Canonical repos confirmed: `jdb78/pytorch-forecasting` **redirects to `sktime/pytorch-forecasting`**
(`gh api repos/jdb78/pytorch-forecasting` returns `"full_name": "sktime/pytorch-forecasting"`).
Darts remains `unit8co/darts`; NeuralForecast remains `Nixtla/neuralforecast`.

| Package | Repo | Latest ver | Release date (PyPI upload) | Releases since 2025-02-02 (18 mo) | Open issues | Open PRs | Commits last 52 wk | Last commit on default branch | `requires_python` | CI Python matrix |
|---|---|---|---|---|---|---|---|---|---|---|
| `pytorch-forecasting` | sktime/pytorch-forecasting (`main`) | 1.8.0 | 2026-06-24T17:04Z | **7** | **627** | **162** | 113 | 2026-08-01T11:09Z | `>=3.10,<3.15` | 3.10, 3.11, 3.12, 3.13, 3.14 × {ubuntu, macos, windows} |
| `darts` | unit8co/darts (`master`) | 0.46.1 | 2026-07-20T10:24Z | **18** | 192 | 23 | 185 | 2026-07-31T08:07Z | `>=3.10` (classifiers list ≤3.12) | 3.10, 3.12 × {ubuntu, macos-14} |
| `neuralforecast` | Nixtla/neuralforecast (`main`) | 3.2.0 | 2026-07-10T13:04Z | **14** | 4 | 4 | 143 | 2026-07-28T20:37Z | `>=3.10` (classifiers ≤3.13) | 3.10, 3.11, 3.12, 3.13 |
| `gluonts` | awslabs/gluonts (`dev`) | 0.17.0 | 2026-07-31T10:57Z | 5 (incl. 1 rc) | 342 | 127 | **23** | 2026-07-31T11:26Z | `>=3.10,<3.15` | not checked |
| `tsai` | timeseriesAI/tsai | 1.0.1 | 2026-05-27T08:26Z | 3 | 30 | – | 35 | 2026-07-23T21:55Z | `>=3.10` | not checked |
| `ludwig` | ludwig-ai/ludwig | 0.17.8 | 2026-07-27T00:40Z | ~10 | 2 | – | 271 | 2026-07-27T21:05Z | `>=3.12` | not checked |
| `pytorchts` | zalandoresearch/pytorch-ts | 0.6.0 | **2022-04-24** | 0 | 104 | – | 0 | **2024-06-14** | `>=3.6` | – |

Release cadence detail (raw, from `gh api repos/<r>/releases`):

- pytorch-forecasting: v1.3.0 2025-02-06, v1.4.0 2025-06-13, v1.5.0 2025-10-10, v1.6.0 2026-01-16,
  v1.6.1 2026-01-23, v1.7.0 2026-04-05, v1.8.0 2026-06-24. Roughly quarterly, accelerating.
- Darts: 0.38.0 2025-10-03 … 0.46.1 2026-07-20 — 12 releases in the last ~10 months alone. Monthly.
- NeuralForecast: v3.0.2 2025-06-17 … v3.2.0 2026-07-10 — monthly-ish.
- GluonTS: 0.16.1 2025-04-08, 0.16.2 2025-06-27, **12-month gap**, 0.16.3 2026-06-29, 0.17.0rc1
  2026-07-22, 0.17.0 2026-07-31. Recently revived, but only 23 commits in 52 weeks.

Effectively abandoned:

- **`zalandoresearch/pytorch-ts` (PyTorchTS): yes, abandoned.** Last PyPI release 2022-04-24, last push
  to the repo 2024-06-14, 104 open issues, 0 commits in the last 52 weeks, `requires_python >=3.6`.
  Not a candidate.
- **Original Google Research TFT (`google-research/google-research/tft`): not a candidate.** It is a
  TF1-era Keras research drop inside a monorepo — `requirements.txt` pins `pandas>=0.25.3`,
  `tensorflow-probability>=0.8.0`; `tft/libs/tft_model.py` builds on `tf.keras.backend.*`. No PyPI
  package, no packaging, no Python 3.12 testing. Useful only as the reference for architectural detail.
- **`tsai`: not a TFT candidate.** `gh api repos/timeseriesAI/tsai/contents/tsai/models` lists no
  TFT (TST, TSTPlus, PatchTST, TSiTPlus, TabTransformer, …). Code search for `TFT` in the repo:
  `total_count: 0`. Maintained but irrelevant.
- **`ludwig`: not a TFT candidate.** Code search for `"temporal fusion"` in `ludwig-ai/ludwig`:
  `total_count: 0`. It is actively released (0.17.8, 2026-07-27, `requires_python >=3.12`) but ships no
  TFT encoder.
- **"Chronos-style" foundation models: out of scope for this problem.** Darts 0.46.1 does ship
  `chronos2_model.py`, `timesfm2p5_model.py`, `tirex_model.py`, `patchtst_fm_model.py`. These are
  pretrained univariate/limited-covariate forecasters; none of them consume ~200 custom
  known-future covariates or ~40 domain categoricals, which is the entire point of this project.
  Not comparable to a trained TFT here.
- **`pytorch-tft` forks**: a GitHub repository search sorted by `updated` for
  `temporal fusion transformer in:name,description` returns only individual coursework/portfolio
  repos (0–2 stars). No maintained standalone TFT library exists outside the four above.

A caveat on the pytorch-forecasting issue count: 627 open issues is by far the worst of the three,
but recent commits are substantive feature work
(`[ENH] Add v2 implementation of DecoderMLP (#2355)` 2026-08-01,
`[ENH] Add feature scaling and normalization to EncoderDecoderDataModule (#2302)` 2026-07-14), and a
`Roadmap 2026` issue (#1993) is open. The backlog reflects an in-progress v2 API migration, not
abandonment. **Could not verify** how many of the 627 are stale/duplicate without manual triage.

---

## 2. Covariate model

**pytorch-forecasting — first class, exact match.** From
[`TimeSeriesDataSet.__init__`](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L439-L470):

```python
static_categoricals: list[str] | None = None,
static_reals: list[str] | None = None,
time_varying_known_categoricals: list[str] | None = None,
time_varying_known_reals: list[str] | None = None,
time_varying_unknown_categoricals: list[str] | None = None,
time_varying_unknown_reals: list[str] | None = None,
```

The docstring for `time_varying_known_reals` reads "list of continuous variables that change over time
and are known in the future". This maps directly onto the project: ~200 astro/calendar features go in
`time_varying_known_*`, and `time_varying_unknown_reals = [target]` (the target's own history) — which
is exactly the degenerate case this project is in, since price-derived features are dropped.

Note the model builds **separate** encoder and decoder VSNs from these lists
(`time_varying_reals_encoder` includes the target, `time_varying_reals_decoder` does not) —
[`_tft.py` L258-283](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py#L258-L283).
So a nearly-empty past-only set is handled correctly, not faked.

**Darts — first class for reals only.** `TFTModel.fit(series, past_covariates=..., future_covariates=...)`.
`future_covariates` are **mandatory** unless `add_relative_index=True` or `add_encoders` supplies them:
`_verify_past_future_covariates` raises when `future_covariates is None and not self.add_relative_index`
([`tft_model.py` L1184-1191](https://github.com/unit8co/darts/blob/0.46.1/darts/models/forecasting/tft_model.py#L1184-L1191)).
The docstring is explicit: "The TFT applies multi-head attention queries on future inputs from mandatory
``future_covariates``." Since almost everything here is known-future, that is fine — but see Q5, the
covariates must all be numeric.

**NeuralForecast — first class naming.** `TFT.__init__(h, input_size, stat_exog_list=None,
hist_exog_list=None, futr_exog_list=None, ...)`
([`tft.py` L577-590](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/models/tft.py#L577-L590)),
with class flags `EXOGENOUS_FUTR = True`, `EXOGENOUS_HIST = True`, `EXOGENOUS_STAT = True`
([L569-571](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/models/tft.py#L569-L571)).
The docstrings say "static **continuous** columns / historic **continuous** columns / future
**continuous** columns" — the split is real, the categoricals are not (Q5).

**GluonTS — first class, and the split is duplicated for cats:** `dynamic_dims` ("Sizes of the
real-valued dynamic features that are known in the future"), `past_dynamic_dims` ("... only known in
the past"), `dynamic_cardinalities`, `past_dynamic_cardinalities`, `static_dims`,
`static_cardinalities`
([`estimator.py` L100-118, L144-202](https://github.com/awslabs/gluonts/blob/v0.17.0/src/gluonts/torch/model/tft/estimator.py#L100-L118)).

---

## 3. Windowing ownership — the crux

**Every one of the four libraries owns windowing.** None of them accepts a pre-built past-60/future-30
3D tensor as its primary input. The hand-rolled tensor pipeline should be dropped.

**pytorch-forecasting.** Required input object: **one long-format `pd.DataFrame`**, one row per
(series, time step), plus a `TimeSeriesDataSet` describing it. `TimeSeriesDataSet` *is* a
`torch.utils.data.Dataset` (`class TimeSeriesDataSet(Dataset)`, L159); `_construct_index` (L1742)
enumerates every legal (start, encoder_len, decoder_len) triple and `__getitem__` (L2109) slices the
window on the fly. `to_dataloader()` (L2542) wraps it.

Escape hatches, in increasing order of divergence:

1. `TimeSeriesDataSet.from_dataset(train_ds, new_df, min_prediction_idx=..., predict=True)` /
   `from_parameters(params, df)` — reuse fitted encoders and scalers on a new dataframe
   (L1643, L1686). `get_parameters()` explicitly carries `categorical_encoders` and `scalers`
   (L1623-1640). **This is the supported walk-forward CV mechanism.**
2. Subclass `TimeSeriesDataSet` and override `__getitem__` / `_construct_index`.
3. **Full bypass**: instantiate `TemporalFusionTransformer(...)` directly — every field
   `from_dataset` would have set (`x_reals`, `x_categoricals`, `embedding_sizes`, `embedding_paddings`,
   `embedding_labels`, `static_categoricals`, `time_varying_reals_encoder`,
   `time_varying_categoricals_decoder`, `hidden_continuous_sizes`, `max_encoder_length`, …) is a plain
   constructor argument (L139-172) — and feed it your own `DataLoader` yielding the batch contract from
   `_collate_fn` (L2525-2540):

   ```python
   (dict(encoder_cat=…, encoder_cont=…, encoder_target=…, encoder_lengths=…,
         decoder_cat=…, decoder_cont=…, decoder_target=…, decoder_lengths=…,
         decoder_time_idx=…, groups=…, target_scale=…),
    (target, weight))
   ```

   This is not "unsupported" so much as "undocumented but stable and explicit". Keep it in reserve.

**Darts.** Required input object: `darts.TimeSeries` (numpy-backed, with a `pd.DatetimeIndex` or
`pd.RangeIndex`), one per target/past-cov/future-cov. `TorchForecastingModel.fit()` calls
`_build_train_dataset()` → `SequentialTorchTrainingDataset`
([`torch_forecasting_model.py` L639-660](https://github.com/unit8co/darts/blob/0.46.1/darts/models/forecasting/torch_forecasting_model.py#L639-L660)).
Darts has the **best-documented escape hatch of the four**: a public
`fit_from_dataset(train_dataset: TorchTrainingDataset, val_dataset=None, ...)` (L1226), and `fit()`'s
own docstring points at it ("... calling :func:`fit_from_dataset()` with a custom
:class:`darts.utils.data.TorchTrainingDataset`"). Sub-classing models are told they "can override
this method to return a custom `TorchTrainingDataset`". So Darts is the one library that would
genuinely consume a hand-rolled window pipeline cleanly — it just fails on categoricals.

**NeuralForecast.** Two levels. The high-level `NeuralForecast(models, freq).fit(df=...)` takes a long
dataframe with `unique_id`, `ds`, `y` + exog columns. The low-level `BaseModel.fit(dataset, val_size,
test_size, ...)` (`_base_model.py` L2161) takes a `neuralforecast.tsdataset.TimeSeriesDataset` — a flat
padded `(total_rows, n_cols)` temporal tensor plus an `indptr` of series boundaries
(`TimeSeriesDataset.from_df`, `tsdataset.py` L355). But windows are **still** cut inside the model by
`_create_windows` (L883) using `unfold`, so you cannot hand it windows either.

**GluonTS.** Input is an iterable of dicts (`{"start": pd.Period, "target": np.ndarray,
"feat_dynamic_real": (D, T) array, "feat_dynamic_cat": (C, T) array, "feat_static_cat": …}`);
`TFTInstanceSplitter` inside the estimator's `_create_instance_splitter` cuts windows.

**Consequence for the sample-tensor ticket:** the artefact is a **long dataframe**, not a
`(N, 90, 200)` tensor. See the FE-constraint section.

---

## 4. Irregular (trading-day) time index

The row set contains only NSE trading days; weekends and market holidays are absent. Concretely:

**pytorch-forecasting — cleanest.** There is **no `freq` parameter anywhere**. `time_idx` is an integer
column, enforced by `_validate_data`:

```python
assert (
    data[self.time_idx].dtype.kind == "i"
), "Timeseries index should be of type integer"
```

([L1000-1003](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L1000-L1003)).
Docs: "integer typed column denoting the time index within `data` … If there are no missing
observations, the time index should increase by `+1` for each subsequent sample."

**Concrete workaround:** `time_idx = df["trading_date"].rank(method="first").astype("int64") - 1`, i.e.
0..N-1 over the sorted trading-day set. That is a dense integer index by construction, so leave
`allow_missing_timesteps=False` (the default). *Do not* set `allow_missing_timesteps=True` — its
documented behaviour is "missing timesteps that are automatically **filled up** … if a specific
timeseries has only samples for 1, 2, 4, 5, the sample for 3 will be generated on-the-fly", which would
manufacture phantom holiday rows if the index were ever calendar-based. Keep the real calendar date as
a **non-feature** metadata column for joins and CV-fold boundaries.

**Darts — workable, via `RangeIndex`.** `TimeSeries.__init__` branches on index type
([`timeseries.py` L286-360](https://github.com/unit8co/darts/blob/0.46.1/darts/timeseries.py#L286-L360)):
a `pd.RangeIndex` is accepted with `freq = times.step` (an integer). A plain integer `pd.Index` is
auto-converted via `_restore_range_indexed`. But a **`pd.DatetimeIndex` of trading days fails**: freq
must be known or inferable, and otherwise

> "The time index is missing the `freq` attribute, and the frequency could not be directly inferred.
> This probably comes from inconsistent date frequencies with missing dates. If you know the actual
> frequency, try setting `fill_missing_dates=True, freq=actual_frequency` …"

Taking that advice with `freq="B"` would insert NaN rows for every NSE holiday — wrong. So Darts also
requires the rank-encoded integer index (`TimeSeries.from_dataframe(df, time_col=None)` on a
`RangeIndex`, or `TimeSeries.from_values`).

**NeuralForecast — explicitly supports an integer index.** `NeuralForecast.__init__(models, freq:
Union[str, int], ...)`; docstring: "Frequency of the data. Must be a valid pandas or polars offset
alias, **or an integer**." `utilsforecast.validation.validate_freq` enforces the pairing:

```python
if _is_int_dtype(times) and not isinstance(freq, int):
    raise ValueError("Time column contains integers but the specified frequency is not an integer. "
                     "Please provide a valid integer, e.g. `freq=1`")
```

So `ds` = integer trading-day rank, `freq=1`. A `CustomBusinessDay` offset would technically be
accepted as a `pd.offsets.BaseOffset` in the `freq` union, but the trading-day gaps come from NSE
holidays, so an integer index is strictly safer. **Could not verify** whether every downstream
`utilsforecast` helper (`offset_times`, forecast-index generation) round-trips a `CustomBusinessDay`
correctly — untested.

**GluonTS.** `start` is a `pd.Period` and the target must be a contiguous array. Use a synthetic daily
`Period` index over the trading-day rank and pass `time_features=[]` so no calendar features are
derived from the (fictional) dates — `time_features` defaults to
`time_features_from_frequency_str(self.freq)` when `None`
([`estimator.py` L204-206](https://github.com/awslabs/gluonts/blob/v0.17.0/src/gluonts/torch/model/tft/estimator.py#L204-L206)),
so passing `[]` explicitly is required.

---

## 5. Categorical embeddings with declared cardinality

This axis is the decisive one.

### pytorch-forecasting — yes, and cardinality can be forced

Mechanism, verified in source:

1. `NaNLabelEncoder(add_nan: bool = False, warn: bool = True)`
   ([`encoders.py` L267-288](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/encoders.py#L267-L288)).
   `fit()` builds `self.classes_` from `np.unique(y)` — i.e. from whatever Series you fit it on, **not
   necessarily the training data**. With `add_nan=True`, `classes_["nan"] = 0` and real levels start
   at 1; `transform()` then maps any unseen value to 0 with a `UserWarning` instead of raising
   (L398-410). With `add_nan=False`, an unknown category raises
   `KeyError("Unknown category '…' encountered. Set `add_nan=True` to allow unknown categories")`.
2. `TimeSeriesDataSet._preprocess_data` **does not re-fit an encoder you supply pre-fitted**:

   ```python
   if name not in self._categorical_encoders:
       self._categorical_encoders[name] = NaNLabelEncoder().fit(data[name])
   elif self._categorical_encoders[name] is not None and name not in self.target_names:
       try:
           check_is_fitted(self._categorical_encoders[name])
       except NotFittedError:
           self._categorical_encoders[name] = self._categorical_encoders[name].fit(data[name])
   ```

   ([L1129-1138](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L1129-L1138)).
3. `TemporalFusionTransformer.from_dataset` sizes the embedding tables from those encoders:

   ```python
   embedding_labels = {name: encoder.classes_ for name, encoder in dataset.categorical_encoders.items() …}
   embedding_sizes = {
       name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
       for name, encoder in dataset.categorical_encoders.items() …
   }
   embedding_sizes.update(kwargs.get("embedding_sizes", {}))
   ```

   ([`_base_model.py` L1996-2007](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/base/_base_model.py#L1996-L2007)).
   The last line means you can **also** override any embedding size explicitly at `from_dataset` time.
4. `MultiEmbedding` then allocates `nn.Embedding(cardinality, emb_dim)` per column, with
   `max_embedding_size=hidden_size`
   ([`_tft.py` L216-220](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py#L216-L220)),
   and `emb_dim = get_embedding_size(n) = min(round(1.6 * n**0.56), 100)` when only a cardinality is
   given ([`utils/_utils.py` L140-159](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/utils/_utils.py#L140-L159)).
   For n=28 that is `round(1.6 * 28**0.56) = 10`, capped at `hidden_size`.

So the recipe is:

```python
encoders = {
    col: NaNLabelEncoder(add_nan=True, warn=False).fit(pd.Series(levels, dtype="object"))
    for col, levels in categories_list.items()
}
ds = TimeSeriesDataSet(df, ..., categorical_encoders=encoders, ...)
# embedding table for `col` now has 1 + len(levels) rows regardless of what the data contains
```

Two gotchas, both source-verified:

- **Categorical columns must be string/object dtype (or a pandas Categorical whose *categories* are
  non-numeric).** `_validate_data` rejects numeric categoricals:
  `ValueError(f"Data type of category {name} was found to be numeric - use a string type / categorified string")`
  ([L1010-1023](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py#L1010-L1023)).
  **The FE pipeline must therefore emit categoricals as strings, not integer codes.**
- **Column names must not contain `.`** — `_validate_data` raises
  `"column names must not contain '.' characters"`.

Related: `embedding_paddings` maps index 0 to a zero vector, and `dropout_categoricals` is derived from
`encoder.add_nan`, which is what makes an unseen level degrade gracefully at inference.

### Darts — no. Static covariates only.

`_TFTModule.__init__(..., categorical_embedding_sizes: dict[str, tuple[int, int]], ...)` feeds exactly
one consumer: `self.static_covariates_vsn`
([`tft_model.py` L151, L177, L547](https://github.com/unit8co/darts/blob/0.46.1/darts/models/forecasting/tft_model.py#L151)).
The docstring is unambiguous: "A dictionary used to construct embeddings for **categorical static
covariates**. The keys are the column names of the categorical static covariates… Note that
``TorchForecastingModels`` only support numeric data."

There is **no path for ~40 time-varying categoricals** in Darts' TFT. You would one-hot them upstream
(28 levels × 40 columns → up to ~1000 extra real covariates, each becoming its own VSN input and its own
GRN — catastrophic for the CPU budget), or ordinal-encode them and lie to the model by declaring them
reals. Both are wrong. **This is why Darts is rejected.**

### NeuralForecast — infrastructure exists, but TFT is excluded

NeuralForecast 3.2.0 added genuine categorical exogenous support: `BaseModel.__init__(...,
cat_exog_list=None, categorical_cardinalities=None, ...)`
([`_base_model.py` L152-153](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/common/_base_model.py#L152-L153)),
`_build_cat_embeddings` allocating `nn.Embedding(cardinality + 1, emb_dim)` with "+1 row reserved for
OOV / unseen categories (index 0)" (L573-582), and per-stream splits
`hist_cat_exog_list` / `futr_cat_exog_list` / `stat_cat_exog_list`.

But this is gated on a class flag, and **TFT does not set it**:

```python
EXOGENOUS_CAT = False  # If the model can embed categorical exogenous variables   (_base_model.py L119)
...
def _check_categorical_exog(self):
    if self.cat_exog_list and not self.EXOGENOUS_CAT:
        raise Exception(f"{type(self).__name__} does not support categorical exogenous variables.")
```

([L548-551](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/common/_base_model.py#L548-L551)).
A GitHub code search for `"EXOGENOUS_CAT = True" repo:Nixtla/neuralforecast` (fetched 2026-08-02)
returns **23 files** — `mlp.py, tcn.py, rnn.py, gru.py, lstm.py, deepar.py, tide.py, xlstm.py, kan.py,
bitcn.py, deepnpts.py, nhits.py, xlinear.py, timesnet.py, timexer.py, informer.py, nbeatsx.py,
tsmixerx.py, mlpmultivariate.py, fedformer.py, dilated_rnn.py, autoformer.py, vanillatransformer.py` —
and **`tft.py` is not among them**. `tft.py` declares only `EXOGENOUS_FUTR/HIST/STAT`
([L569-571](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/models/tft.py#L569-L571)).
Passing `cat_exog_list` to `NeuralForecast`'s `TFT` raises. **This is why NeuralForecast is rejected.**

Even if TFT gained the flag, the declared-cardinality semantics would be weaker than
pytorch-forecasting's. `_build_categorical_vocab` builds the mapping from **observed** uniques and
treats the declared cardinality only as an upper bound:

```python
uniques = sorted(frame[col].dropna().unique().tolist())
if len(uniques) > max_card:
    raise ValueError(f"Categorical feature '{col}' has {len(uniques)} distinct values … but "
                     f"`categorical_cardinalities` declares only {max_card}. Increase the declared cardinality.")
self.categorical_vocab_[col] = {val: i + 1 for i, val in enumerate(uniques)}
```

([`core.py` L333-354](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/core.py#L333-L354)),
and `_encode_categoricals` maps everything else to 0 ("unseen -> 0"). So the embedding table is padded
to `cardinality + 1` rows, but declared-yet-unobserved levels all **collapse into the single OOV
bucket** rather than each getting a reserved row. No crash at inference — but no per-level parameter
either.

### GluonTS — yes, and the cleanest declaration

`dynamic_cardinalities: Optional[List[int]]` ("Cardinalities of the categorical dynamic features that
are known in the future") and `past_dynamic_cardinalities` are passed straight into the model config as
`"c_feat_dynamic_cat": self.dynamic_cardinalities`
([`estimator.py` L156-157, L391-394](https://github.com/awslabs/gluonts/blob/v0.17.0/src/gluonts/torch/model/tft/estimator.py#L156-L157)).
Nothing is inferred from the data — you state the cardinality and it is honoured. This is the single
best API on this axis; it just comes attached to the least-maintained package.

---

## 6. Multi-horizon output

The TFT paper (Lim, Arık, Loeff, Pfister, *Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting*, [arXiv:1912.09363](https://arxiv.org/abs/1912.09363), IJF 37(4) 2021) is a
**direct** multi-horizon quantile model: it emits all τ ∈ {1..τ_max} in a single forward pass and is
trained with the quantile loss, with no autoregressive feedback. All four libraries preserve this.

**pytorch-forecasting.** `TemporalFusionTransformer(..., output_size: int | list[int] = 7,
loss: MultiHorizonMetric = None, max_encoder_length: int = 10, ...)`; the default is
`loss = QuantileLoss()` with 7 quantiles
([`_tft.py` L139-206](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py#L139-L206)).
Horizon comes from `TimeSeriesDataSet(max_prediction_length=10)`; the forward pass emits
`(batch, 10, output_size)` in one shot. Nothing is recursive. Available losses (from
`pytorch_forecasting/metrics/`): `QuantileLoss`, `MAE`, `RMSE`, `MAPE`, `SMAPE`, `MASE`, `PoissonLoss`,
`TweedieLoss`, `CrossEntropy`, and distributional (`NormalDistributionLoss`,
`LogNormalDistributionLoss`, `NegativeBinomialDistributionLoss`, `MultivariateNormalDistributionLoss`,
`BetaDistributionLoss`, `ImplicitQuantileNetworkDistributionLoss`, `MQF2DistributionLoss`),
plus `MultiLoss` for multi-target.

For this project: `max_prediction_length=10`, `loss=QuantileLoss(quantiles=[...])` (or `MAE` for a point
model). **The target is ONE column** — the daily log return `r_t = log(C_t/C_{t-1})` — and the decoder's
10 steps *are* `r_1..r_10`. Do not model 10 separate target columns.

**Note on the 60/30 window.** TFT consumes known-future covariates for exactly
`max_encoder_length + max_prediction_length` steps. With `max_encoder_length=60` and
`max_prediction_length=10`, only 10 of the 30 future steps are used. If the extra 20 steps of
astro-lookahead are believed to carry signal, either (a) set `max_prediction_length=30` and score only
the first 10 (cheap: +28% VSN FLOPs, see Q7), or (b) fold the t+11..t+30 values in as *additional
known-real columns at time t* (e.g. `moon_phase_lead_15`). Option (b) is cheaper but inflates the
variable count, which is the expensive axis. **Open question for the prototype.**

**Darts.** Direct within `output_chunk_length`, which is also "the number of future values from future
covariates to use as a model input". `predict(n)` is one-shot when `n <= output_chunk_length` and
**becomes autoregressive when `n > output_chunk_length`** — so set `output_chunk_length=10` and never
call `predict(n>10)`. Default `likelihood=QuantileRegression`; set `likelihood=None, loss_fn=<nn.Module>`
for a deterministic model.

**NeuralForecast.** `RECURRENT = False  # If the model produces forecasts recursively (True) or direct
(False)` ([`tft.py` L573-575](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/models/tft.py#L573-L575)).
`h=10`. Default `loss=MAE()`; the loss library is the richest of the four —
`MAE, MSE, RMSE, MAPE, SMAPE, MASE, relMSE, QuantileLoss, MQLoss, IQLoss, HuberLoss, HuberQLoss,
HuberMQLoss, HuberIQLoss, TukeyLoss, DistributionLoss, PMM/GMM/NBMM, ISQF, sCRPS, FreDF, Accuracy`
([`losses/pytorch.py`](https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/losses/pytorch.py)).

**GluonTS.** `prediction_length=10`, `quantiles` defaulting to `[0.1 … 0.9]`, `distr_output` defaulting
to `QuantileOutput`. Direct.

---

## 7. CPU viability

### Where the cost is

From the pytorch-forecasting source, per **encoder** Variable Selection Network with `n_r` reals,
`n_c` categoricals, `hidden_size = H`, `hidden_continuous_size = h_c`:

- per real variable: a `nn.Linear(1, h_c)` prescaler **plus** a full
  `GatedResidualNetwork(h_c, min(h_c,H), H)`;
- per categorical variable: only a `ResampleNorm(emb_dim, H)` — layer-norm + a learned gate, **no
  matmul**. Categoricals are nearly free;
- one "flattened" GRN over the concatenation of every variable:
  `GatedResidualNetwork(Σ input_sizes, min(H, n), n, context_size=H)`

([`sub_modules.py` L281-345](https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.py#L281-L345)).
The forward pass is a **Python loop over variables** (L363-382), so with 200 variables each VSN call
dispatches ~1,600 small tensor ops.

Interpretable multi-head attention over 70–90 steps at `d_model=16` is negligible by comparison.
The paper's own framing (§4.5, arXiv:1912.09363) is that the VSN is what gives interpretability; it is
also what costs.

### Arithmetic (this project's shapes)

Parameter and MAC counts computed directly from the module definitions above, `enc=60`, `dec=10`,
`n_r=161` reals (160 numeric + the target) and `n_c=40` categoricals at ~12 levels each,
6500 samples/epoch, backward ≈ 2× forward:

| Config | Params | Fwd MACs/sample (VSN / LSTM / attn) | Train GFLOP/epoch |
|---|---|---|---|
| **Set 1**, `H=16, h_c=8`, dec=10 | ~254 k | 6.61 M / 0.14 M / 0.35 M | **277** |
| Set 1, `H=32, h_c=16`, dec=10 | ~856 k | 24.8 M / 0.57 M / 1.13 M | 1,034 |
| Set 1, `H=16, h_c=8`, dec=**30** | ~254 k | 8.49 M / 0.18 M / 0.45 M | 356 |
| **Set 2 (PCA)**, 30 PCs + 40 cats, `H=16, h_c=8` | ~73 k | 1.59 M / 0.14 M / 0.35 M | **81** |
| Set 2 (PCA), `H=32, h_c=16` | ~231 k | 5.46 M / 0.57 M / 1.13 M | 279 |

Read across: the VSN is **93 %** of forward MACs at set-1/H=16, and its cost scales linearly in the
number of *real* variables. Cutting 160 numeric features to ~30 principal components cuts total training
FLOPs by **~3.4×** (277 → 81 GFLOP/epoch). **Yes, PCA set 2 is materially cheaper than set 1, and the
reason is precisely that the VSN builds one GRN per real variable.** It also cuts the op-dispatch count
by ~4×, which on CPU matters more than the FLOPs.

### Wall-clock estimate

Two independent bounds, both landing in the same place for set 1 at `H=16`:

- **FLOP bound.** 277 GFLOP/epoch. These are hundreds of tiny GEMMs (e.g. `(64,60,8) @ (8,8)`), so
  effective throughput on an 8–16 core desktop CPU is far below peak — call it 5–20 GFLOP/s.
  → **~15–55 s/epoch**.
- **Dispatch bound.** ~161 prescalers + 161 GRNs in the encoder VSN and ~160 in the decoder VSN;
  each GRN is ~7 kernel launches. ≈ 2,600 tensor ops per forward, ≈ 7,800 per training step including
  backward. At ~10–30 µs per small CPU op that is **~0.08–0.23 s/step**; at batch 64 there are ~102
  steps/epoch → **~8–24 s/epoch**.

**Order of magnitude: tens of seconds per epoch; ~10–60 min for a 50–100-epoch fit on set 1; roughly
3× less on PCA set 2.** A walk-forward CV with 8 folds is therefore a **~1.5–8 hour** job for set 1 and
under 2 hours for set 2. That is tractable without a GPU.

Corroborating evidence (**anecdotal**, labelled as such): pytorch-forecasting issue
[#349 "GPU Training is slower than CPU training"](https://github.com/sktime/pytorch-forecasting/issues/349)
— the maintainer's reply is "Question of the bottleneck. If your model is small and the batch size is
also not too big, transferring data to the GPU might cost more time than evaluating on the CPU." This
matches the dispatch-bound analysis: at this model size CPU is genuinely competitive, and the absence
of a GPU is not the constraint people assume. I found **no published CPU benchmark** for TFT at these
shapes; the numbers above are derived, not measured.

**I could not run a live benchmark in this sandbox.** `uv pip install pytorch-forecasting==1.8.0`
succeeded, but importing torch failed with
`OSError: Error relocating .../libgomp-a34b3233.so.1: pthread_attr_setaffinity_np: symbol not found` —
this sandbox runs **musl libc 1.2.6**, and PyTorch ships only manylinux (glibc) wheels. If the real
target box is also musl/Alpine, **torch will not install at all** and that is a blocker independent of
library choice. Verify the target box is glibc before anything else.

### Concrete config advice

- `hidden_size=16`, `hidden_continuous_size=8`, `attention_head_size=1`, `lstm_layers=1`,
  `dropout=0.1..0.3`. `hidden_continuous_size` is the single biggest FLOP lever for reals — it is the
  input width of all 160 per-variable GRNs. Do not raise it before you have raised `hidden_size`.
- `share_single_variable_networks=True` shares the per-variable GRNs between encoder and decoder —
  ~halves VSN parameters (254 k → ~135 k), which matters a lot for overfitting on 6500 samples.
  It does *not* reduce FLOPs.
- Batch size 64–128. Larger batches amortise the per-op dispatch overhead, which is the real cost.
- `torch.set_num_threads(n_physical_cores)` and `Trainer(accelerator="cpu", devices=1)`. Do **not**
  use `num_workers>0` in the dataloader unless you measure a win — `TimeSeriesDataSet.__getitem__` is
  cheap and worker IPC often costs more.
- Consider `torch.compile` on the LightningModule; the VSN's per-variable loop is exactly the pattern
  that benefits from graph fusion. **Unverified** — prototype it.
- Prefer **set 2 (PCA)** for hyperparameter search and CV sweeps, then confirm the winner on set 1.
- Drop `logging_metrics` to `nn.ModuleList([MAE()])` — the default computes SMAPE/MAE/RMSE/MAPE every
  step.

---

## Constraint on the FE pipeline's output format

**This is the binding constraint for the sample-tensor ticket.**

### The FE pipeline emits a LONG DATAFRAME. It does NOT emit windows.

`pytorch-forecasting`'s `TimeSeriesDataSet` owns windowing. Emitting a pre-built
`(N, 90, 200)` tensor would be thrown away.

### Required schema

One row per trading day, one column per feature. Exactly one file/artefact per input set
(set 1 = raw features, set 2 = PCA), each a `pd.DataFrame` (parquet on disk).

| Column | dtype | Role | Notes |
|---|---|---|---|
| `series_id` | `str` (constant, e.g. `"NIFTY50"`) | `group_ids=["series_id"]` | Required even for a single series. Must be a string, not an int. |
| `time_idx` | `int64` | `time_idx=` | **Dense rank of the trading day, 0..N-1**, strictly `+1` per row. Assertion `dtype.kind == "i"` is enforced. |
| `trade_date` | `datetime64[ns]` | metadata only | **Not** a model feature. Used to define CV fold boundaries / purge+embargo windows and to join predictions back. Must be excluded from every `time_varying_*` list. |
| `log_ret` | `float32` | `target=` | `log(C_t / C_{t-1})`. **One column, not ten.** The decoder's 10 steps are `r_1..r_10`. |
| `<numeric astro feature>` × ~160 | `float32` | `time_varying_known_reals` | No NaNs (impute upstream; TFT does not tolerate NaN in reals). Names must not contain `.`. |
| `<categorical astro feature>` × ~40 | **`object` (Python `str`) or `pd.Categorical` with string categories** | `time_varying_known_categoricals` | **Must NOT be integer codes.** `_validate_data` raises `"Data type of category X was found to be numeric - use a string type / categorified string"`. Emit the human-readable level label. |
| `pc_0 … pc_29` (set 2 only) | `float32` | `time_varying_known_reals` | PCA is per-timestep and numeric-only; the 90-step time axis is *not* collapsed, so PCA output is still one row per trading day. |

Additional contract points:

- **Sorted** by `time_idx` ascending, no duplicates, no gaps.
- **No lookahead in the row itself**: for a row at `time_idx = t`, every astro/calendar column must be
  the value *at* t. Future-ness is expressed by *membership in `time_varying_known_reals`*, not by
  shifting columns. The library does the shifting when it builds decoder windows.
- **No `.` in any column name.**
- Set 1 and set 2 must share identical `series_id`, `time_idx`, `trade_date`, `log_ret` and the same
  ~40 categorical columns — only the numeric block differs.

### Alongside the dataframe: the declared-cardinality artefact

The FE pipeline must also emit (or pass through unchanged) `categories_list.json`:

```json
{
  "nakshatra": ["Ashwini", "Bharani", "Krittika", "..."],
  "tithi":     ["Shukla_Pratipada", "..."],
  "moon_sign": ["Mesha", "Vrishabha", "..."]
}
```

— a mapping `column name -> ordered list of ALL declared string levels`, including levels never observed
in the data. This is injected at `TimeSeriesDataSet` construction time, **not** into the dataframe:

```python
from pytorch_forecasting.data.encoders import NaNLabelEncoder

categorical_encoders = {
    col: NaNLabelEncoder(add_nan=True, warn=False).fit(pd.Series(levels, dtype="object"))
    for col, levels in json.load(open("categories_list.json")).items()
}
```

`add_nan=True` reserves index 0 for genuinely-unknown values; every declared level (observed or not)
gets its own row, and `from_dataset` sizes the embedding table as `len(encoder.classes_)`.

### Known-future vs past-only split

Expressed **entirely in the `TimeSeriesDataSet` constructor**, not in the data:

```python
TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="log_ret",
    group_ids=["series_id"],
    max_encoder_length=60,
    min_encoder_length=60,
    max_prediction_length=10,
    min_prediction_length=10,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_reals=["time_idx", *NUMERIC_ASTRO_COLS],     # or pc_0..pc_29 for set 2
    time_varying_known_categoricals=CATEGORICAL_ASTRO_COLS,
    time_varying_unknown_reals=["log_ret"],                          # target history only
    time_varying_unknown_categoricals=[],
    categorical_encoders=categorical_encoders,
    target_normalizer=GroupNormalizer(groups=["series_id"]),         # or None / TorchNormalizer
    allow_missing_timesteps=False,                                   # index is already dense
    add_relative_time_idx=True,
    add_target_scales=False,
    randomize_length=None,
)
```

### Worked example of the emitted rows

```
series_id  time_idx  trade_date  log_ret   sun_long  moon_long  ...  nakshatra   tithi              moon_sign
NIFTY50    0         2005-01-03   0.00412   282.31    114.87    ...  "Ashlesha"  "Krishna_Ashtami"  "Karka"
NIFTY50    1         2005-01-04  -0.00187   283.29    128.44    ...  "Magha"     "Krishna_Navami"   "Simha"
NIFTY50    2         2005-01-05   0.00033   284.27    142.10    ...  "Magha"     "Krishna_Dashami"  "Simha"
...
```
(note: `2005-01-01` and `2005-01-02` are a weekend and simply do not exist as rows; `time_idx` does not
skip.)

dtypes: `series_id: object`, `time_idx: int64`, `trade_date: datetime64[ns]`,
`log_ret: float32`, numeric astro: `float32`, categorical astro: `object`/`category[str]`.

### What the FE pipeline must NOT do

- **Must not pre-window.** No `(N, 60, F)` / `(N, 30, F)` / `(N, 90, 200)` tensors. The library slices
  windows in `__getitem__`.
- **Must not emit 10 target columns.** One `log_ret` column; the horizon is a model hyperparameter.
- **Must not integer-encode categoricals.** Emit strings. `TimeSeriesDataSet` rejects numeric
  categoricals, and integer codes would silently desynchronise from the declared level list.
- **Must not one-hot categoricals.** That would blow up the VSN (Q7) and defeat the embeddings.
- **Must not reindex onto a calendar frequency, insert holiday rows, or forward-fill across
  non-trading days.**
- **Must not shift/lead the known-future columns itself.** Membership in `time_varying_known_reals`
  is the mechanism.
- **Must not scale/normalise the target.** `target_normalizer` belongs to `TimeSeriesDataSet` so the
  same fitted transform is reused via `from_dataset` on validation/test folds. Feature scaling for
  reals may be pre-applied *or* delegated to the dataset's `scalers=`; **pick one and record it** —
  delegating is safer for walk-forward CV because `get_parameters()` carries fitted scalers forward.
- **Must not drop `trade_date`** — the CV harness needs real dates to enforce purge/embargo.

### If the decision is later reversed

Should a future prototype prove the dataframe route unworkable, the fallback is *not* Darts — it is
pytorch-forecasting's constructor bypass (Q3, escape hatch 3): the FE pipeline would then emit
the `_collate_fn` batch dict (`encoder_cat`, `encoder_cont`, `decoder_cat`, `decoder_cont`,
`encoder_lengths`, `decoder_lengths`, `decoder_time_idx`, `groups`, `target_scale`, `encoder_target`,
`decoder_target`) with `encoder_cat`/`decoder_cat` as `int64` embedding indices consistent with a
hand-built level→index map. Do not build toward this now.

---

## Open questions / verify by prototype

1. **glibc vs musl on the real training box.** The research sandbox is musl (Alpine); PyTorch ships
   manylinux/glibc wheels only and `import torch` fails there. Confirm the target box before anything.
2. **Real wall-clock.** The Q7 numbers are derived from source-level parameter/MAC arithmetic, not
   measured. Run one epoch on set 2 at `hidden_size=16, batch_size=64` and calibrate.
3. **10 vs 30 decoder steps.** Whether to set `max_prediction_length=30` (and score the first 10) so the
   full 30-step known-future astro window is consumed, versus `max_prediction_length=10` plus explicit
   lead-features. Costs +28 % VSN FLOPs; unknown value.
4. **`torch.compile` on the VSN loop.** Plausible large win on CPU; unverified.
5. **Feature scaling ownership** — pre-scale in FE vs `TimeSeriesDataSet(scalers=...)`. The latter
   round-trips through `get_parameters()`/`from_parameters()` for CV folds, but with ~200 columns the
   per-column `sklearn` transformers add fit cost. Measure.
6. **`min_prediction_idx` for purge/embargo.** `from_dataset(..., min_prediction_idx=k)` is the intended
   mechanism for producing a validation split, but it filters at subsequence level
   (`x[time_idx] >= min_prediction_idx - max_encoder_length - max_lag`). Verify by hand that no encoder
   window in a validation fold reaches back across the embargo boundary.
7. **Whether the 627 open pytorch-forecasting issues include anything that bites the v1 TFT path
   specifically.** Not triaged. Issue #1825 ("[BUG] TFT + categorical features seems not to be
   compatible with DDP in some situations") is open but DDP-only, so irrelevant on a single CPU box.
8. **NeuralForecast may add `EXOGENOUS_CAT = True` to TFT.** The feature landed for 23 other models in
   this release cycle (`[FEAT] Categorical feature support for distributed workflows (#1581)`,
   2026-07-28). If TFT gains it, NeuralForecast becomes a live alternative — but the observed-uniques
   vocab semantics would still be weaker than a pre-fitted `NaNLabelEncoder`.
9. **Whether `pd.CustomBusinessDay` round-trips through NeuralForecast/`utilsforecast`** — not tested;
   moot under the recommendation.

---

## Sources

GitHub API (all fetched 2026-08-02):
- https://api.github.com/repos/sktime/pytorch-forecasting
- https://api.github.com/repos/jdb78/pytorch-forecasting (redirects to sktime/pytorch-forecasting)
- https://api.github.com/repos/sktime/pytorch-forecasting/releases
- https://api.github.com/repos/sktime/pytorch-forecasting/commits
- https://api.github.com/repos/sktime/pytorch-forecasting/stats/participation
- https://api.github.com/repos/unit8co/darts
- https://api.github.com/repos/unit8co/darts/releases
- https://api.github.com/repos/unit8co/darts/commits
- https://api.github.com/repos/unit8co/darts/stats/participation
- https://api.github.com/repos/Nixtla/neuralforecast
- https://api.github.com/repos/Nixtla/neuralforecast/releases
- https://api.github.com/repos/Nixtla/neuralforecast/commits
- https://api.github.com/repos/Nixtla/neuralforecast/stats/participation
- https://api.github.com/repos/awslabs/gluonts
- https://api.github.com/repos/awslabs/gluonts/releases
- https://api.github.com/repos/timeseriesAI/tsai
- https://api.github.com/repos/timeseriesAI/tsai/contents/tsai/models
- https://api.github.com/repos/ludwig-ai/ludwig
- https://api.github.com/repos/ludwig-ai/ludwig/releases
- https://api.github.com/repos/ludwig-ai/ludwig/contents/ludwig/encoders
- https://api.github.com/repos/zalandoresearch/pytorch-ts
- https://api.github.com/search/issues (open issue / open PR counts per repo)
- https://api.github.com/search/code?q=%22EXOGENOUS_CAT+%3D+True%22+repo:Nixtla/neuralforecast
- https://api.github.com/search/code?q=%22temporal+fusion%22+repo:ludwig-ai/ludwig
- https://api.github.com/search/repositories?q=temporal+fusion+transformer+in:name,description&sort=updated

PyPI JSON API (fetched 2026-08-02):
- https://pypi.org/pypi/pytorch-forecasting/json
- https://pypi.org/pypi/darts/json
- https://pypi.org/pypi/neuralforecast/json
- https://pypi.org/pypi/gluonts/json
- https://pypi.org/pypi/tsai/json
- https://pypi.org/pypi/ludwig/json
- https://pypi.org/pypi/pytorchts/json

Library source (read at the pinned release tag):
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/timeseries/_timeseries_v2.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/data/encoders.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/_tft_v2.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/nn/embeddings.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/models/base/_base_model.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pytorch_forecasting/utils/_utils.py
- https://github.com/sktime/pytorch-forecasting/blob/v1.8.0/pyproject.toml
- https://github.com/sktime/pytorch-forecasting/blob/main/.github/workflows/test.yml
- https://github.com/unit8co/darts/blob/0.46.1/darts/models/forecasting/tft_model.py
- https://github.com/unit8co/darts/blob/0.46.1/darts/models/forecasting/torch_forecasting_model.py
- https://github.com/unit8co/darts/blob/0.46.1/darts/timeseries.py
- https://github.com/unit8co/darts/blob/0.46.1/pyproject.toml
- https://github.com/unit8co/darts/blob/master/.github/workflows/merge.yml
- https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/models/tft.py
- https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/common/_base_model.py
- https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/core.py
- https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/tsdataset.py
- https://github.com/Nixtla/neuralforecast/blob/v3.2.0/neuralforecast/losses/pytorch.py
- https://github.com/Nixtla/neuralforecast/blob/main/.github/workflows/pytest.yml
- https://github.com/Nixtla/utilsforecast/blob/main/utilsforecast/validation.py
- https://github.com/awslabs/gluonts/blob/v0.17.0/src/gluonts/torch/model/tft/estimator.py
- https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
- https://github.com/google-research/google-research/blob/master/tft/requirements.txt

Official docs:
- https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
- https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html

Issue tracker (anecdotal, labelled as such in text):
- https://github.com/sktime/pytorch-forecasting/issues/349
- https://github.com/sktime/pytorch-forecasting/issues/1825
- https://github.com/sktime/pytorch-forecasting/issues/1993
- https://github.com/unit8co/darts/issues/3148

Paper:
- https://arxiv.org/abs/1912.09363 — Lim, Arık, Loeff, Pfister, "Temporal Fusion Transformers for
  Interpretable Multi-horizon Time Series Forecasting", International Journal of Forecasting 37(4), 2021
