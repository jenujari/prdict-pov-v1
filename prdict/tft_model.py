"""Build set 1/set 2's TFT datasets and run the fixed hyperparameter search (#39).

Mirrors `prdict/xgboost_model.py`'s shape: two dataset builders (set 1
fold-invariant, set 2 fold-scoped PCA), a `fit_one`/`random_search`/
`run_context` ladder, and a `Progress` sink a caller can subclass for live
feedback. The contract itself (architecture, search space, budget, selection/
refit rule) is fixed in `kb/tft_spec.json` by `scripts/build_tft_spec.py`,
committed before this module's first real fit; this module only ever reads
that file.

The walk-forward mechanics ride entirely on `TimeSeriesDataSet`/`from_dataset`
(#11's `dataset.tft_frame`/`make_dataset`/`build_fold_views`), verified by hand
against `kb/fold_spec.json`'s real fold boundaries (see `main()`):

  - A **training** dataset is built from a frame sliced to exactly the block's
    origins (`[block[0] - (past-1), block[-1] + future]` in `time_idx`) — the
    slice itself is what bounds the enumerable origins, `TimeSeriesDataSet` has
    no direct "origin list" parameter.
  - A **prediction** dataset is built via `TimeSeriesDataSet.from_dataset(training,
    frame_sliced_to_upper_bound_only, predict=False, min_prediction_idx=...)`.
    Critically, the frame passed here must **not** be lower-bounded — the
    library's own `min_prediction_idx` filter only computes correctly relative
    to a frame that still starts at (or before) the true encoder start; slicing
    the front off breaks its internal row accounting (verified by hand, not
    assumed — see `main()`'s origin-alignment check).

    uv run python -m prdict.tft_model    # self-check: build both sets' fold-1 datasets
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from prdict import encoding, set1, set2
from prdict.dataset import (
    ELAPSED,
    SERIES,
    SERIES_ID,
    TARGET,
    TIME_IDX,
    Covariates,
    _elapsed_series,
    _target_series,
    build_fold_views,
    encoded_features,
    make_dataset,
)
from prdict.trading_calendar import TradingCalendar

ROOT = Path(__file__).resolve().parent.parent
TFT_SPEC_PATH = ROOT / "kb" / "tft_spec.json"


def load_tft_spec(path: Path = TFT_SPEC_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/build_tft_spec.py first.")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# covariate restriction — shared by set 1 and set 2
# --------------------------------------------------------------------------


def restricted_covariates(spec: encoding.EncodingSpec, feature_columns: list[str]) -> Covariates:
    """Split an arbitrary feature list into known reals/categoricals.

    `feature_columns` may include set 2's synthetic `pca_*` names, which have
    no encoding-spec family membership — anything **not** in the `categorical`
    family rides as a real (numeric, boolean, or PCA alike), exactly like
    `dataset.covariates()`'s split of the full 280-column block (only the
    `categorical` family has `tft_role="time_varying_known_categorical"`).
    `elapsed` is always appended, same as `dataset.covariates()`.
    """
    cat_family = set(spec.family("categorical").columns)
    known_cats = [c for c in feature_columns if c in cat_family]
    known_reals = [c for c in feature_columns if c not in cat_family]
    known_reals.append(ELAPSED)
    return Covariates(
        time_varying_known_reals=known_reals,
        time_varying_known_categoricals=known_cats,
        time_varying_unknown_reals=[],
        time_varying_unknown_categoricals=[],
    )


def _make_dataset_restricted(
    frame: pd.DataFrame,
    cal: TradingCalendar,
    spec: encoding.EncodingSpec,
    globals_: encoding.GlobalEncoders,
    columns: list[str],
    **overrides,
):
    """`dataset.make_dataset`, with covariates restricted to `columns`.

    `make_dataset`'s own `**overrides` merge onto its computed kwargs last, so
    passing the restricted lists here fully replaces the full-280 defaults —
    no change to `dataset.py` needed.
    """
    from pytorch_forecasting.data.encoders import TorchNormalizer

    cov = restricted_covariates(spec, columns)
    kwargs = dict(
        time_varying_known_reals=cov.time_varying_known_reals,
        time_varying_known_categoricals=cov.time_varying_known_categoricals,
        time_varying_unknown_reals=[],
        time_varying_unknown_categoricals=[],
        scalers={col: None for col in cov.time_varying_known_reals},
        target_normalizer=TorchNormalizer(method="identity"),
    )
    kwargs.update(overrides)
    return make_dataset(frame, cal, spec, globals_, **kwargs)


# --------------------------------------------------------------------------
# frame builders — set 1 fold-invariant, set 2 fold-scoped
# --------------------------------------------------------------------------


def set1_columns(spec: encoding.EncodingSpec) -> list[str]:
    return set1.load_spec().features_for("tft")


def set2_columns() -> list[str]:
    return set2.features_for("tft")


def _extended_book(cal: TradingCalendar) -> pd.DataFrame:
    """Book columns over the FULL trading-day index (history + forward), not
    just history.

    `dataset.tft_frame()` stops at `cal.n_history` (needs `y` observed) — fine
    for training, but a late holdout origin's *decoder* only needs known-future
    covariates (#3), not an observed label, so bounding the frame to history
    starves it of rows it's entitled to. Concretely: the holdout's last
    trainable origin is only 10 sessions before the last observed close (#8),
    but the decoder needs 30 — the ~20-origin gap ADR0002 documents as a
    *training*-pool effect turned out to also truncate `predict_origins` for
    the holdout arm (caught via a real failed run, not anticipated up front).
    `y` is fabricated (0) past history, same convention as `tft_frame()`'s
    row-0 — never read as a real target, since `model.predict()` doesn't use
    decoder-row targets, only known covariates.
    """
    n_total = len(cal.sessions)
    y = np.zeros(n_total, dtype=np.float32)
    y[: cal.n_history] = _target_series(cal)
    return pd.DataFrame(
        {
            TIME_IDX: np.arange(n_total, dtype=np.int32),
            SERIES_ID: pd.Categorical([SERIES] * n_total),
            TARGET: y,
            ELAPSED: _elapsed_series(cal).astype(np.float32),
        }
    )


def build_set1_frame(
    cal: TradingCalendar, spec: encoding.EncodingSpec, globals_: encoding.GlobalEncoders, fold: Any
) -> pd.DataFrame:
    """Set 1's fold-scoped scaled frame, over the full history+forward index
    (see `_extended_book`). `linear_numeric` scaled to `fold`'s training rows
    (#11's `build_fold_views`, unchanged). Covariate *lists* passed to dataset
    construction are what actually restrict this down to set 1's 90 columns.
    """
    feats = encoded_features(cal, spec, globals_, "tft")
    book = _extended_book(cal)
    frame = pd.concat([book, feats[spec.all_columns].reset_index(drop=True)], axis=1)
    _state, scaled = build_fold_views(spec, frame, fold, cal)
    return scaled


def build_set2_frame_for_fold(
    cal: TradingCalendar, spec: encoding.EncodingSpec, globals_: encoding.GlobalEncoders, fold: Any
) -> pd.DataFrame:
    """Set 2's fold-scoped frame over the full history+forward index: `pca_1..48`
    replace `linear_numeric`, book columns prepended (`_extended_book`) — mirrors
    `dataset.tft_frame`'s construction, using `set2.fit_fold`'s transform instead
    of the raw encoded features. Must be rebuilt per fold/final_train; the PCA
    loadings are fold-specific (#13).
    """
    feats = encoded_features(cal, spec, globals_, "tft")
    state = set2.fit_fold(spec, feats, fold, cal)
    transformed = state.transform(feats)

    book = _extended_book(cal)
    frame = pd.concat([book, transformed.reset_index(drop=True)], axis=1)
    assert frame[TIME_IDX].dtype.kind == "i"
    return frame


# --------------------------------------------------------------------------
# walk-forward dataset construction — verified against real fold boundaries
# --------------------------------------------------------------------------


def _slice_upper(frame: pd.DataFrame, cal: TradingCalendar, last_origin: pd.Timestamp) -> pd.DataFrame:
    hi = cal.position(last_origin) + cal.future
    return frame[frame[TIME_IDX] <= hi].reset_index(drop=True)


def training_dataset(
    frame: pd.DataFrame,
    cal: TradingCalendar,
    spec: encoding.EncodingSpec,
    globals_: encoding.GlobalEncoders,
    columns: list[str],
    origins: pd.DatetimeIndex,
):
    """A `TimeSeriesDataSet` whose enumerable origins are exactly `origins`.

    The frame slice — `[origins[0] - (past-1), origins[-1] + future]` in
    `time_idx` — is what bounds the origin set; verified in `main()` against
    `fold.inner_train`/`fold.train` directly, not assumed from the library docs.
    """
    lo = cal.position(origins[0]) - (cal.past - 1)
    hi = cal.position(origins[-1]) + cal.future
    sliced = frame[(frame[TIME_IDX] >= lo) & (frame[TIME_IDX] <= hi)].reset_index(drop=True)
    return _make_dataset_restricted(sliced, cal, spec, globals_, columns)


def prediction_dataset(training_ds, frame: pd.DataFrame, cal: TradingCalendar, origins: pd.DatetimeIndex):
    """A `TimeSeriesDataSet` sharing `training_ds`'s fitted encoders/scalers,
    enumerable origins exactly `origins`.

    `frame` must **not** be lower-bounded (see module docstring) — only sliced
    at the upper end, past `origins[-1]`'s decoder tail.
    """
    sliced = _slice_upper(frame, cal, origins[-1])
    min_prediction_idx = cal.position(origins[0]) + 1
    from pytorch_forecasting import TimeSeriesDataSet

    return TimeSeriesDataSet.from_dataset(
        training_ds, sliced, predict=False, min_prediction_idx=min_prediction_idx, stop_randomization=True
    )


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    model: Any  # TemporalFusionTransformer
    best_epoch: int
    best_val_loss: float
    params: dict


class Progress:
    """No-op progress sink; callers that want live feedback pass a subclass.

    `on_epoch` fires after every training epoch of every fit (epoch index,
    `max_epochs`) — the only sub-fit-granularity signal Lightning's `Trainer`
    gives easily. `on_trial` fires once a search trial's early stopping has
    settled. Neither return value is used; a subclass observes and renders, it
    never influences training.
    """

    def on_epoch(self, epoch: int, max_epochs: int) -> None: ...
    def on_trial(self, trial_idx: int, n_trials: int, result: "FitResult") -> None: ...


_NULL_PROGRESS = Progress()


def _build_model(training_ds, params: dict, tft_spec: dict):
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import MAE

    fa = tft_spec["fixed_architecture"]
    return TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"],
        dropout=params["dropout"],
        hidden_continuous_size=fa["hidden_continuous_size"],
        lstm_layers=fa["lstm_layers"],
        share_single_variable_networks=fa["share_single_variable_networks"],
        loss=MAE(),
        logging_metrics=torch.nn.ModuleList([MAE()]),
        learning_rate=tft_spec["core"]["learning_rate"],
    )


class _BestEpochTracker:
    """Tracks the epoch with the lowest `val_loss` seen so far, independent of
    when `EarlyStopping` actually stops training (which fires `patience`
    epochs *after* the best one, not on it) — this is what `best_epoch` in
    `FitResult` should mean: the model the refit phase re-trains for exactly
    this many epochs, not however long early stopping happened to run.
    """

    def __init__(self) -> None:
        self.best_epoch = 0
        self.best_loss = float("inf")

    def callback(self):
        from lightning.pytorch.callbacks import Callback

        tracker = self

        class _Tracker(Callback):
            def on_validation_end(self, trainer, pl_module):
                loss = trainer.callback_metrics.get("val_loss")
                if loss is None:
                    return
                loss = float(loss)
                if loss < tracker.best_loss:
                    tracker.best_loss = loss
                    tracker.best_epoch = trainer.current_epoch

        return _Tracker()


def fit_one(
    training_ds,
    val_ds,
    params: dict,
    tft_spec: dict,
    *,
    max_epochs: int | None = None,
    patience: int | None = None,
    checkpoint_dir: Path | None = None,
    resume_ckpt: Path | None = None,
    progress: Progress = _NULL_PROGRESS,
) -> FitResult:
    """One TFT fit. With `val_ds`, early-stops on its loss (the search phase),
    tracking the actual best epoch (not the epoch training stopped at — early
    stopping only fires `patience` epochs after the best one). Without
    `val_ds`, trains for exactly `max_epochs` with no early stopping (the
    full-block refit phase) — the same two-phase shape #38 uses for XGBoost.
    """
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

    es = tft_spec["early_stopping"]
    max_epochs = max_epochs if max_epochs is not None else es["max_epochs"]
    torch.set_num_threads(tft_spec["cpu"]["num_threads"])

    model = _build_model(training_ds, params, tft_spec)

    class _EpochProgress(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            progress.on_epoch(trainer.current_epoch, max_epochs)

    callbacks: list = [_EpochProgress()]
    if checkpoint_dir is not None:
        callbacks.append(ModelCheckpoint(dirpath=str(checkpoint_dir), save_last=True, save_top_k=0))

    tracker = _BestEpochTracker() if val_ds is not None else None
    if val_ds is not None:
        callbacks.append(tracker.callback())
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience if patience is not None else es["patience"]))

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=tft_spec["cpu"]["accelerator"],
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=checkpoint_dir is not None,
        callbacks=callbacks,
    )

    train_dl = training_ds.to_dataloader(train=True, batch_size=tft_spec["batch_size"], num_workers=tft_spec["cpu"]["num_workers"])
    val_dl = val_ds.to_dataloader(train=False, batch_size=tft_spec["batch_size"], num_workers=tft_spec["cpu"]["num_workers"]) if val_ds is not None else None

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    if val_ds is not None:
        best_epoch = tracker.best_epoch
        best_val_loss = tracker.best_loss
    else:
        best_epoch = trainer.current_epoch
        best_val_loss = float("nan")

    return FitResult(model=trainer.model, best_epoch=best_epoch, best_val_loss=best_val_loss, params=params)


def _sample_trials(tft_spec: dict) -> list[dict]:
    """The fixed `n_trials` draws from `kb/tft_spec.json`'s space, fixed seed."""
    s = tft_spec["search"]
    rng = np.random.default_rng(s["seed"])
    space = s["space"]
    keys = list(space)
    return [{k: rng.choice(space[k]).item() for k in keys} for _ in range(s["n_trials"])]


def random_search(training_ds, val_ds, tft_spec: dict, progress: Progress = _NULL_PROGRESS) -> FitResult:
    """Fit every trial on the inner split, return the one with the best inner-val loss."""
    trials = _sample_trials(tft_spec)
    results = []
    for i, params in enumerate(trials):
        r = fit_one(training_ds, val_ds, params, tft_spec, progress=progress)
        results.append(r)
        progress.on_trial(i, len(trials), r)
    return min(results, key=lambda r: r.best_val_loss)


# --------------------------------------------------------------------------
# per-context run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    context: str
    set_name: str
    predictions: pd.DataFrame
    winning_params: dict
    best_epoch: int
    inner_val_loss: float
    n_train: int
    n_predict: int
    seconds: float


def run_context(
    cal: TradingCalendar,
    spec: encoding.EncodingSpec,
    globals_: encoding.GlobalEncoders,
    tft_spec: dict,
    *,
    context: str,
    set_name: str,
    train_origins: pd.DatetimeIndex,
    inner_train_origins: pd.DatetimeIndex,
    inner_val_origins: pd.DatetimeIndex,
    predict_origins: pd.DatetimeIndex,
    fold_or_final: Any,
    checkpoint_dir: Path | None = None,
    progress: Progress = _NULL_PROGRESS,
) -> RunResult:
    """Search on the inner split, refit on the whole block, predict on `predict_origins`."""
    start = time.monotonic()

    columns = set1_columns(spec) if set_name == "set1" else set2_columns()
    frame = (
        build_set1_frame(cal, spec, globals_, fold_or_final)
        if set_name == "set1"
        else build_set2_frame_for_fold(cal, spec, globals_, fold_or_final)
    )

    ds_inner_train = training_dataset(frame, cal, spec, globals_, columns, inner_train_origins)
    ds_inner_val = prediction_dataset(ds_inner_train, frame, cal, inner_val_origins)
    search = random_search(ds_inner_train, ds_inner_val, tft_spec, progress)

    ds_full_train = training_dataset(frame, cal, spec, globals_, columns, train_origins)
    refit = fit_one(
        ds_full_train,
        None,
        search.params,
        tft_spec,
        max_epochs=max(search.best_epoch, 1),
        checkpoint_dir=checkpoint_dir,
        progress=progress,
    )

    ds_predict = prediction_dataset(ds_full_train, frame, cal, predict_origins)
    predict_dl = ds_predict.to_dataloader(train=False, batch_size=tft_spec["batch_size"], num_workers=tft_spec["cpu"]["num_workers"])
    result = refit.model.predict(predict_dl, mode="prediction", return_index=True)
    output = result.output.numpy()[:, : cal.horizon]
    origins = cal.sessions[result.index["time_idx"].to_numpy() - 1]

    predictions = pd.DataFrame(
        output, index=pd.DatetimeIndex(origins, name="origin"),
        columns=[f"y_pred_{k}" for k in range(1, output.shape[1] + 1)],
    ).loc[predict_origins]

    return RunResult(
        context=context,
        set_name=set_name,
        predictions=predictions,
        winning_params=search.params,
        best_epoch=search.best_epoch,
        inner_val_loss=search.best_val_loss,
        n_train=len(train_origins),
        n_predict=len(predict_origins),
        seconds=time.monotonic() - start,
    )


def main() -> None:
    from prdict.folds import load_fold_spec
    from prdict.trading_calendar import load_calendar

    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)
    fs = load_fold_spec()
    fold1 = fs.folds(cal)[0]

    c1 = set1_columns(spec)
    c2 = set2_columns()
    print(f"set1 tft columns: {len(c1)}  set2 tft columns: {len(c2)}")
    assert len(c1) == 90, f"set1 tft columns should be 90, got {len(c1)}"
    assert len(c2) == 129, f"set2 tft columns should be 129, got {len(c2)}"

    frame1 = build_set1_frame(cal, spec, globals_, fold1)
    ds_train = training_dataset(frame1, cal, spec, globals_, c1, fold1.inner_train)

    def origins_of(ds) -> pd.DatetimeIndex:
        t = ds.index["time"].to_numpy() + (cal.past - 1)
        return pd.DatetimeIndex(cal.sessions[t]).sort_values()

    assert origins_of(ds_train).equals(fold1.inner_train), "training dataset origins must equal fold.inner_train exactly"
    print(f"set1 fold1 training dataset: {len(ds_train)} samples, origins match fold.inner_train exactly")

    ds_val = prediction_dataset(ds_train, frame1, cal, fold1.inner_val)
    assert origins_of(ds_val).equals(fold1.inner_val), "prediction dataset origins must equal fold.inner_val exactly"
    print(f"set1 fold1 inner-val dataset: {len(ds_val)} samples, origins match fold.inner_val exactly — purge boundary verified")

    frame2 = build_set2_frame_for_fold(cal, spec, globals_, fold1)
    ds_train2 = training_dataset(frame2, cal, spec, globals_, c2, fold1.inner_train)
    assert origins_of(ds_train2).equals(fold1.inner_train)
    print(f"set2 fold1 training dataset: {len(ds_train2)} samples, origins match fold.inner_train exactly")

    tft_spec = load_tft_spec()
    print(f"tft spec: loss={tft_spec['core']['loss']}  search {tft_spec['search']['n_trials']} trials")


if __name__ == "__main__":
    main()
