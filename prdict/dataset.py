"""The dataframe contract both models eat, and the two physical layouts on top of it.

Resolves wayfinder ticket #11 (retargeted by #5, decided at the 2026-08-08
realignment — see `docs/adr/0001`, `docs/adr/0002`). The encoding spec (#9) fixes
*semantics* — what each family becomes and where its transform is fitted. This
module fixes *layout*: the single long-format frame, the four covariate lists
that map it onto a `TimeSeriesDataSet`, and the flat design matrix XGBoost eats.

Two models, one encoded source of truth, two views of it:

  * **TFT** — `pytorch_forecasting.TimeSeriesDataSet` *is* the windower. It slices
    the encoder-60 / decoder-30 windows itself in `__getitem__`, so there is no
    hand-rolled tensor here — only the long frame (one row per trading session)
    and the covariate lists. `max_encoder_length=60`, `max_prediction_length=30`;
    the decoder spans all 30 known-future sessions and is scored on the first 10.

  * **XGBoost** — no windower, so this module flattens. One flat row per origin,
    every one of the 90 window sessions times all 280 features ≈ 25,200 columns,
    plus `elapsed_1..10`. Built **per fold** (the whole-history matrix is ~650 MB
    float32), after encoding, before #12's in-fold prune.

Alignment is the whole ticket, so `main()` verifies the library's own windowing
against the raw closes at one hand-checked origin (item 3).

    uv run python -m prdict.dataset    # self-check against nft50.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prdict import angles, columns, encoding
from prdict.encoding import EncodingSpec, GlobalEncoders
from prdict.targets import close_series, elapsed_days, step_returns
from prdict.trading_calendar import CSV_PATH, TradingCalendar, load_calendar, restrict

# The frame carries three bookkeeping columns beside the 280 features. They are
# named once here so no downstream stage invents its own spelling.
TIME_IDX = "time_idx"
SERIES_ID = "series_id"
TARGET = "y"
ELAPSED = "elapsed"
SERIES = "nifty50"  # one instrument, so the group id is a constant.


# --------------------------------------------------------------------------
# item 2 — the four covariate lists
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Covariates:
    """How the encoded families map onto `TimeSeriesDataSet`'s four lists.

    Per #3 **every astronomical feature is known-future** — the ephemeris is
    computable for the decoder block exactly as for the encoder — so the two
    *unknown* lists are empty but for the target itself. That is stated here, not
    left implied: the only thing the model may not see ahead of the origin is the
    return it is predicting, which `TimeSeriesDataSet` carries through `target=`,
    never through `time_varying_unknown_reals`.
    """

    time_varying_known_reals: list[str]
    time_varying_known_categoricals: list[str]
    time_varying_unknown_reals: list[str]
    time_varying_unknown_categoricals: list[str]


def covariates(spec: EncodingSpec) -> Covariates:
    """Populate the four lists from the encoding spec's per-family `tft_role`.

    `elapsed` is appended to the known reals by hand: it is a decoder-side
    known-future covariate (D4) that lives beside the target, **not** a member of
    the 280-column feature block, so #9's width assertion must never see it.
    """
    known_reals: list[str] = []
    known_cats: list[str] = []
    for family in spec.families:
        if family.tft_role == "time_varying_known_real":
            known_reals.extend(family.columns)
        elif family.tft_role == "time_varying_known_categorical":
            known_cats.extend(family.columns)
        else:  # pragma: no cover - guards a spec that grew an unknown role
            raise ValueError(
                f"family {family.name!r} has tft_role {family.tft_role!r}; #3 says "
                "every feature is known-future, so only the two known roles are valid"
            )
    known_reals.append(ELAPSED)
    return Covariates(
        time_varying_known_reals=known_reals,
        time_varying_known_categoricals=known_cats,
        time_varying_unknown_reals=[],  # the target is the only unknown real
        time_varying_unknown_categoricals=[],
    )


# --------------------------------------------------------------------------
# the encoded feature frame — shared by both views, over *all* sessions
# --------------------------------------------------------------------------


def encoded_features(
    cal: TradingCalendar,
    spec: EncodingSpec,
    globals_: GlobalEncoders,
    view: str,
    csv_path=CSV_PATH,
) -> pd.DataFrame:
    """The 280 encoded features on the trading-day index, ascending, per view.

    Spans **every** session (history *and* forward): the features are known-future
    (#3), so the forward rows are real astronomical values, not padding. XGBoost's
    flatten reaches 30 sessions past a late origin into that forward block, which
    is exactly why the matrix is built here over the full index rather than over
    history alone.

    `restrict()` sorts ascending, so row `i` is `cal.sessions[i]` — the frame's
    positional index *is* the calendar position, which is what keeps `time_idx`
    honest below.
    """
    cspec = columns.load_spec()
    aspec = angles.load_spec()
    raw = pd.read_csv(csv_path, parse_dates=[cspec.key])
    feat = angles.transform(columns.apply_structural_fills(raw, cspec), aspec)
    feat = restrict(feat, cal, key=cspec.key)
    typed = encoding.apply_dtypes(feat, spec, view, globals_)
    return typed


# --------------------------------------------------------------------------
# item 1 + item 3 — the long frame the TFT eats
# --------------------------------------------------------------------------


def _target_series(cal: TradingCalendar, closes: pd.Series | None = None) -> np.ndarray:
    """The one-session log return landing on each history session, `y[0] = 0`.

    `step_returns` drops the first session (no predecessor); we keep the row so
    that `time_idx` stays equal to `cal.position` for every session, and fill its
    return with 0. That fabricated value is never a *target* — the first trainable
    origin is the 60th session — it only rides in the target-history channel of
    the very earliest encoder windows.
    """
    ret = step_returns(cal, closes)
    y = np.zeros(cal.n_history, dtype=np.float32)
    y[1:] = ret.to_numpy(dtype=np.float32)
    return y


def _elapsed_series(cal: TradingCalendar) -> np.ndarray:
    """Calendar days each step spans, co-indexed with the target, `elapsed[0] = 0`.

    `elapsed[i]` is the span of the step that *lands on* session `i`, exactly like
    `y[i]`, so the covariate and the return it explains share an index. Session 0
    has no incoming step, hence 0.
    """
    span = elapsed_days(cal)
    e = np.zeros(len(cal.sessions), dtype=np.int16)
    e[1:] = span.to_numpy(dtype=np.int16)
    return e


def tft_frame(
    cal: TradingCalendar,
    spec: EncodingSpec,
    globals_: GlobalEncoders,
    closes: pd.Series | None = None,
) -> pd.DataFrame:
    """The long-format frame `TimeSeriesDataSet` consumes — one row per session.

    Restricted to **history** (`y` needs an observed close). The TFT's 30-step
    training origins reach at most to the last observed close, so nothing the
    decoder needs falls off the end. `time_idx` is a dense integer rank of the
    trading days (0..N-1): the calendar has no missing sessions by construction,
    so `allow_missing_timesteps=False` holds and nothing needs a fill.

    Columns: `time_idx` (int32), `series_id` (category, constant), `y` (float32
    target), `elapsed` (float32 known-future covariate), and the 280 features
    typed for the `tft` view.
    """
    feats = encoded_features(cal, spec, globals_, "tft").iloc[: cal.n_history]
    n = cal.n_history
    book = pd.DataFrame(
        {
            TIME_IDX: np.arange(n, dtype=np.int32),
            SERIES_ID: pd.Categorical([SERIES] * n),
            TARGET: _target_series(cal, closes),
            ELAPSED: _elapsed_series(cal)[:n].astype(np.float32),
        }
    )
    # One concat rather than 280 inserts — keeps the typed feature columns and
    # avoids a fragmented frame.
    frame = pd.concat([book, feats[spec.all_columns].reset_index(drop=True)], axis=1)

    assert frame[TIME_IDX].dtype.kind == "i", "time_idx must be integer for TimeSeriesDataSet"
    return frame


def make_dataset(
    frame: pd.DataFrame,
    cal: TradingCalendar,
    spec: EncodingSpec,
    globals_: GlobalEncoders,
    *,
    predict_mode: bool = False,
    **overrides,
):
    """Construct the TFT `TimeSeriesDataSet` from the long frame and covariate lists.

    Encoder = `past` (60), decoder = `future` (30); the caller scores the first
    `horizon` (10). Scaling is **not** delegated to the library: `linear_numeric`
    is fold-scaled in `prdict.encoding.build_fold` before the frame arrives here,
    so every real's internal scaler is disabled to avoid a second, dataset-wide
    (and therefore leaking) fit. The categorical encoders are the pre-fitted
    global `NaNLabelEncoder`s, so no fold can size an embedding table from what it
    happened to observe (map decision 6).
    """
    from pytorch_forecasting import TimeSeriesDataSet

    cov = covariates(spec)
    reals = cov.time_varying_known_reals + cov.time_varying_unknown_reals
    kwargs = dict(
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[SERIES_ID],
        max_encoder_length=cal.past,
        min_encoder_length=cal.past,
        max_prediction_length=cal.future,
        min_prediction_length=cal.future,
        time_varying_known_reals=cov.time_varying_known_reals,
        time_varying_known_categoricals=cov.time_varying_known_categoricals,
        time_varying_unknown_reals=cov.time_varying_unknown_reals,
        time_varying_unknown_categoricals=cov.time_varying_unknown_categoricals,
        categorical_encoders=globals_.label_encoders(),
        scalers={col: None for col in reals},
        allow_missing_timesteps=False,
        add_relative_time_idx=False,
        predict_mode=predict_mode,
    )
    kwargs.update(overrides)
    return TimeSeriesDataSet(frame, **kwargs)


# --------------------------------------------------------------------------
# XGBoost — the full-flatten design matrix, built per fold
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatMatrix:
    """One fold's flat design matrix, plus what XGBoost needs to read it.

    `values` is float32 with the categoricals carried as integer codes; the
    matching `feature_types` list marks those columns `"c"` so a `DMatrix` built
    with `enable_categorical=True` treats them as categorical rather than ordinal.
    """

    origins: pd.DatetimeIndex
    values: np.ndarray  # (n_origins, n_features) float32
    feature_names: list[str]
    feature_types: list[str]  # "c" for categorical columns, "q" otherwise

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)


def _window_offsets(cal: TradingCalendar) -> range:
    """The 90 session offsets a window covers, `t-59 .. t+30` (origin at 0)."""
    return range(-(cal.past - 1), cal.future + 1)


def flatten_columns_for(
    cal: TradingCalendar, columns: list[str], categorical: set[str]
) -> tuple[list[str], list[str]]:
    """The flat matrix's column names and types for an arbitrary column subset.

    Order is offset-major, feature-minor (`col@t-59 … col@t+30`), which is exactly
    the C-order flatten of a `(90, len(columns))` window used below. `elapsed_1..10`
    follow the window block: they are the decoder-side known-future covariate for
    the 10 scored steps (D4), never one of the windowed feature columns.

    Generic core behind `flatten_columns()` (fixed at the full 280-column block)
    and set 1/set 2's smaller column lists (#38) — one implementation of "how a
    window's column names are laid out" rather than a second copy per caller.
    """
    names: list[str] = []
    types: list[str] = []
    for offset in _window_offsets(cal):
        tag = f"@t{offset:+d}"
        for col in columns:
            names.append(col + tag)
            types.append("c" if col in categorical else "q")
    for k in range(1, cal.horizon + 1):
        names.append(f"{ELAPSED}_{k}")
        types.append("q")
    return names, types


def flatten_columns(cal: TradingCalendar, spec: EncodingSpec) -> tuple[list[str], list[str]]:
    """The full 280-column flat matrix's column names and types — fold-invariant."""
    return flatten_columns_for(cal, spec.all_columns, set(spec.family("categorical").columns))


def codes_from_frame(frame: pd.DataFrame, columns: list[str], categorical: set[str]) -> np.ndarray:
    """Pull a `(n_rows, len(columns))` float32 array out of any typed frame.

    Category-dtype columns become their integer codes (never widened to one-hot);
    everything else converts directly. Generic core behind `_feature_codes()` (the
    full 280-column block) and any other typed frame — set 2's fold-fitted PCA
    frame in particular, whose `pca_1..pca_k` columns don't exist in `spec` at all.
    """
    out = np.empty((len(frame), len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        series = frame[col]
        if col in categorical:
            out[:, j] = series.cat.codes.to_numpy(dtype=np.float32)
        else:
            out[:, j] = series.to_numpy(dtype=np.float32)
    return out


def _feature_codes(
    cal: TradingCalendar, spec: EncodingSpec, globals_: GlobalEncoders
) -> np.ndarray:
    """The `(n_sessions, 280)` feature matrix as float32, categoricals as codes.

    Spans every session so the flatten can reach into the forward block. Columns
    follow `spec.all_columns`, the same order `flatten_columns` names.
    """
    feats = encoded_features(cal, spec, globals_, "xgboost")
    return codes_from_frame(feats, spec.all_columns, set(spec.family("categorical").columns))


def flatten_matrix(
    cal: TradingCalendar,
    columns: list[str],
    categorical: set[str],
    codes: np.ndarray,
    elapsed: np.ndarray,
    origins: pd.DatetimeIndex,
) -> FlatMatrix:
    """Flatten each origin's 90-session window of `codes` into one flat row.

    For origin `t`, the window is `t-59 … t+30` (encoder 60 + decoder 30), every
    session's `columns` laid out offset-major, then `elapsed_1..10`. Generic core
    behind `flatten()` (the full 280-column block, built once and fold-invariant)
    and set 1/set 2's smaller, and for set 2 fold-scoped, column subsets (#38) —
    `codes`/`elapsed` are always supplied by the caller here, since which frame
    they come from (raw encoded features vs. a fold-fitted PCA transform) is a
    decision only the caller can make.
    """
    window = cal.past + cal.future  # 90
    positions = cal.sessions.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("an origin is not a trading session")
    starts = positions - (cal.past - 1)
    if starts.min() < 0 or (positions + cal.future).max() >= len(cal.sessions):
        raise ValueError("an origin's window falls outside the calendar")

    # sliding_window_view puts the window on the last axis: (n-89, n_cols, 90).
    win = np.lib.stride_tricks.sliding_window_view(codes, window, axis=0)
    picked = win[starts]  # (n_origins, n_cols, 90)
    block = np.transpose(picked, (0, 2, 1)).reshape(len(origins), window * codes.shape[1])

    steps = np.arange(1, cal.horizon + 1)
    elapsed_block = elapsed[positions[:, None] + steps[None, :]].astype(np.float32)

    values = np.hstack([block.astype(np.float32, copy=False), elapsed_block])
    names, types = flatten_columns_for(cal, columns, categorical)
    if values.shape[1] != len(names):
        raise ValueError(f"flatten produced {values.shape[1]} columns, named {len(names)}")
    return FlatMatrix(origins=origins, values=values, feature_names=names, feature_types=types)


def flatten(
    cal: TradingCalendar,
    spec: EncodingSpec,
    globals_: GlobalEncoders,
    origins: pd.DatetimeIndex,
    *,
    codes: np.ndarray | None = None,
    elapsed: np.ndarray | None = None,
) -> FlatMatrix:
    """Flatten each origin's 90-session window into one flat XGBoost row, full 280 cols.

    The heavy per-session matrix and the elapsed series are fold-invariant; pass
    them in via `codes`/`elapsed` to flatten many folds without rebuilding them.
    """
    if codes is None:
        codes = _feature_codes(cal, spec, globals_)
    if elapsed is None:
        elapsed = _elapsed_series(cal)
    return flatten_matrix(
        cal, spec.all_columns, set(spec.family("categorical").columns), codes, elapsed, origins
    )


# --------------------------------------------------------------------------
# C3 — the folds.py <-> encoding.py wiring
# --------------------------------------------------------------------------


def fold_fit_rows(fold, cal: TradingCalendar) -> pd.Index:
    """Session row labels a fold's scaler may be fitted on.

    The fold's training origins run `train_start … train_end`; the sessions this
    training set actually reads span the earliest encoder start (`train_start-59`)
    through the last origin (`train_end`). Decoder-future sessions are deliberately
    excluded — they reach across the 30-session purge toward the validation block,
    and while the encoder overlap there is harmless (every feature is known-future,
    PR #34), fitting scaler statistics stops at the origin so the boundary is
    unarguable. Returns positional labels into the `tft_frame` (`time_idx`).
    """
    start = cal.position(fold.train[0]) - (cal.past - 1)
    end = cal.position(fold.train[-1])
    return pd.Index(range(max(start, 0), end + 1))


def build_fold_views(spec, frame, fold, cal):
    """Fit the fold-scoped scaler and return the scaled TFT frame (C3 wiring).

    This is the single call the training loop makes per fold: it names the rows
    the scaler may see (`fold_fit_rows`) and hands them to
    `encoding.build_fold` — the only connection between `prdict.folds` and
    `prdict.encoding`, which neither module made for itself. The returned
    `FoldState` refuses to pickle, so it cannot outlive the fold.
    """
    rows = fold_fit_rows(fold, cal)
    state = encoding.build_fold(spec, frame, rows)
    return state, state.transform(frame)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def main() -> None:
    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)

    # C3: the fit boundary is asserted once at pipeline start, structurally.
    encoding.assert_fit_boundary(spec)
    print("fit boundary asserted (encoding.py) — wiring live")

    cov = covariates(spec)
    print(
        f"\ncovariate lists: known {len(cov.time_varying_known_reals)} reals "
        f"(incl. {ELAPSED}) + {len(cov.time_varying_known_categoricals)} categoricals; "
        f"unknown {len(cov.time_varying_unknown_reals)} reals "
        f"(target {TARGET!r} carried separately)"
    )
    assert cov.time_varying_unknown_reals == []
    assert cov.time_varying_unknown_categoricals == []
    assert ELAPSED in cov.time_varying_known_reals
    assert ELAPSED not in spec.all_columns, "elapsed must not be a member of the 280-col block"
    assert len(cov.time_varying_known_reals) == 14 + 92 + 107 + 1
    assert len(cov.time_varying_known_categoricals) == 67

    frame = tft_frame(cal, spec, globals_)
    print(
        f"\ntft frame: {frame.shape[0]} rows x {frame.shape[1]} cols  "
        f"time_idx {frame[TIME_IDX].iloc[0]}..{frame[TIME_IDX].iloc[-1]}  "
        f"dtype {frame[TIME_IDX].dtype}"
    )
    assert frame[TIME_IDX].dtype.kind == "i"
    assert frame[TIME_IDX].is_monotonic_increasing
    assert (frame[TIME_IDX].diff().dropna() == 1).all(), "time_idx must be dense (no gaps)"

    # item 3 — alignment, hand-checked against the raw closes at one origin, and
    # then against the library's own windowing.
    closes = close_series(cal)
    origin = cal.trainable_origins(reach=cal.future)[len(cal.trainable_origins(reach=cal.future)) // 2]
    t = cal.position(origin)
    for k in range(1, cal.horizon + 1):
        expected = float(np.log(closes.iloc[t + k] / closes.iloc[t + k - 1]))
        got = float(frame[TARGET].iloc[t + k])
        assert np.isclose(got, expected), (k, got, expected)
    print(f"\nalignment: frame y[t+k] = log(C_t+k/C_t+k-1) verified at origin {origin.date()}")

    # Identity target normaliser so the decoder tensor carries the raw returns —
    # the check is about *which* timesteps land in the decoder, not their scaling
    # (#38 owns the normaliser). scalers are already disabled in make_dataset.
    from pytorch_forecasting.data.encoders import TorchNormalizer

    ds = make_dataset(
        frame, cal, spec, globals_, target_normalizer=TorchNormalizer(method="identity")
    )
    idx = ds.index
    match = idx.index[idx["index_end"] == t + cal.future]
    sample = ds[int(match[0])]
    decoder_target = sample[1][0].numpy().reshape(-1)[: cal.horizon]
    expected = frame[TARGET].to_numpy()[t + 1 : t + 1 + cal.horizon]
    assert np.allclose(decoder_target, expected, atol=1e-5), (decoder_target, expected)
    print(
        f"  library windowing agrees: TimeSeriesDataSet decoder target at origin "
        f"matches frame y[t+1..t+{cal.horizon}] (encoder {cal.past}, decoder {cal.future})"
    )

    # item 6 / XGBoost — the flat matrix, built per fold, memory measured.
    from prdict.folds import load_fold_spec

    fs = load_fold_spec()
    fold0 = fs.folds(cal)[0]
    state, scaled = build_fold_views(spec, frame, fold0, cal)
    print(
        f"\nC3 fold 0: scaler fit on {state.n_train} session rows, "
        f"{len(state.columns)} linear_numeric columns scaled"
    )
    train_block = scaled.loc[fold_fit_rows(fold0, cal), state.columns].to_numpy(float)
    assert abs(float(np.nanmean(train_block))) < 1e-5
    assert abs(float(np.nanstd(train_block)) - 1.0) < 1e-5
    print("  train block standardised: mean~0 std~1")

    codes = _feature_codes(cal, spec, globals_)
    fm = flatten(cal, spec, globals_, fold0.train, codes=codes)
    per_origin = fm.values.shape[1]
    full = per_origin * len(cal.trainable_origins()) * 4
    print(
        f"\nxgboost flatten (fold 0 train): {fm.values.shape[0]} origins x {per_origin} cols  "
        f"= {fm.nbytes / 1e6:.1f} MB float32"
    )
    print(
        f"  columns: {cal.past + cal.future} sessions x {len(spec.all_columns)} features "
        f"+ {cal.horizon} elapsed = {per_origin}"
    )
    print(f"  whole-history matrix would be {full / 1e6:.0f} MB — hence per-fold, not once")
    assert per_origin == (cal.past + cal.future) * len(spec.all_columns) + cal.horizon
    assert fm.feature_types.count("c") == 67 * (cal.past + cal.future)

    # Flatten alignment: the scored elapsed columns must equal the target's own.
    from prdict.targets import build as build_target

    tgt = build_target(cal)
    common = fold0.train.intersection(tgt.origins)
    fm_row = {o: i for i, o in enumerate(fm.origins)}
    tg_row = {o: i for i, o in enumerate(tgt.origins)}
    probe = common[len(common) // 2]
    flat_elapsed = fm.values[fm_row[probe], -cal.horizon :]
    assert np.allclose(flat_elapsed, tgt.elapsed[tg_row[probe]]), "flatten elapsed misaligned"
    print(f"  elapsed_1..{cal.horizon} matches targets.elapsed at {probe.date()}")


if __name__ == "__main__":
    main()
