"""Build set 1/set 2's flat matrices and run XGBoost's fixed hyperparameter search.

Resolves the reusable half of wayfinder ticket #38 — everything `scripts/
train_xgboost.py` needs but that isn't specific to *which* fold or set is being
trained right now. The contract itself (`multi_strategy`, the search space, the
budget) is fixed in `kb/xgboost_spec.json` by `scripts/build_xgboost_spec.py`,
committed before this module's first real fit; this module only ever reads that
file, never redefines it.

Two matrix builders, because set 1 and set 2 have different fit scopes:

  - **Set 1** (`build_set1_matrix`) is fold-*invariant* — its column list is
    fixed (map decision 4 doesn't apply to a discrete column choice the way it
    does to a fitted scaler), and XGBoost needs no scaling. Built once over
    every trainable origin; folds just slice rows out of it.
  - **Set 2** (`build_set2_matrix_for_fold`) is fold-*scoped* — the PCA loadings
    genuinely differ per fold (#13), so it must be rebuilt for every fold and
    for the final holdout refit.

Both matrices use the **`xgboost` encoding view** specifically, not `tft` —
`weekday` (the one categorical column either set carries) must stay pandas
`category` dtype all the way through so `dataset.codes_from_frame` can read its
`.cat.codes`; the `tft` view would give it a plain string instead.

    uv run python -m prdict.xgboost_model    # self-check: build both matrices for fold 1
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from prdict import encoding, independence, set1, set2
from prdict.dataset import FlatMatrix, codes_from_frame, encoded_features, flatten_matrix
from prdict.trading_calendar import TradingCalendar

ROOT = Path(__file__).resolve().parent.parent
XGB_SPEC_PATH = ROOT / "kb" / "xgboost_spec.json"


def load_xgboost_spec(path: Path = XGB_SPEC_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/build_xgboost_spec.py first.")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# matrix builders
# --------------------------------------------------------------------------


def _elapsed_for_flatten(cal: TradingCalendar) -> np.ndarray:
    """The same per-session elapsed series `dataset.flatten()` uses internally."""
    from prdict.dataset import _elapsed_series

    return _elapsed_series(cal)


def build_set1_matrix(
    cal: TradingCalendar,
    spec: encoding.EncodingSpec,
    globals_: encoding.GlobalEncoders,
    origins: pd.DatetimeIndex,
) -> FlatMatrix:
    """Set 1's flat matrix: fixed 10-column list, no fold-scoped fitting.

    XGBoost doesn't need scaling — trees are invariant to monotone rescaling of
    a split variable — so unlike set 2, this can be built once and sliced per
    fold rather than refit.
    """
    columns = set1.load_spec().features_for("xgboost")
    categorical = {c for c in columns if c in set(spec.family("categorical").columns)}

    feats = encoded_features(cal, spec, globals_, "xgboost")
    codes = codes_from_frame(feats, columns, categorical)
    elapsed = _elapsed_for_flatten(cal)
    return flatten_matrix(cal, columns, categorical, codes, elapsed, origins)


def build_set2_matrix_for_fold(
    cal: TradingCalendar,
    spec: encoding.EncodingSpec,
    globals_: encoding.GlobalEncoders,
    fold: Any,
    origins: pd.DatetimeIndex,
) -> FlatMatrix:
    """Set 2's flat matrix for one fold: fit PCA on `fold`'s rows, then flatten.

    `fold` is duck-typed exactly like `set2.fit_fold`/`dataset.fold_fit_rows`
    expect — a walk-forward `Fold` or a `SimpleNamespace(train=...)` for the
    final holdout refit. Must be rebuilt per fold; the PCA loadings are
    fold-specific by design (#13).
    """
    categorical_cols = independence.load_spec().categorical_columns_for("xgboost")
    columns = set2.load_spec().pca_columns() + categorical_cols
    categorical = set(categorical_cols)

    frame = encoded_features(cal, spec, globals_, "xgboost")
    state = set2.fit_fold(spec, frame, fold, cal)
    transformed = state.transform(frame)

    codes = codes_from_frame(transformed, columns, categorical)
    elapsed = _elapsed_for_flatten(cal)
    return flatten_matrix(cal, columns, categorical, codes, elapsed, origins)


def _target_matrix(cal: TradingCalendar, origins: pd.DatetimeIndex) -> np.ndarray:
    """The 10-step scored target, aligned to `origins` — never the 30-step extension."""
    from prdict.targets import build as build_target

    tgt = build_target(cal)
    row = {o: i for i, o in enumerate(tgt.origins)}
    missing = [o for o in origins if o not in row]
    if missing:
        raise ValueError(f"{len(missing)} origins have no scored target, e.g. {missing[0].date()}")
    return tgt.y[[row[o] for o in origins]]


def slice_matrix(fm: FlatMatrix, origins: pd.DatetimeIndex) -> FlatMatrix:
    """Rows of `fm` for `origins`, in `origins`' order — set 1's one-build-many-slices path."""
    row = {o: i for i, o in enumerate(fm.origins)}
    idx = [row[o] for o in origins]
    return FlatMatrix(
        origins=origins,
        values=fm.values[idx],
        feature_names=fm.feature_names,
        feature_types=fm.feature_types,
    )


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    booster: xgb.Booster
    best_iteration: int
    best_rmse: float
    params: dict


class Progress:
    """No-op progress sink; callers that want live feedback pass a subclass.

    `on_round` fires after every boosting round of every fit (round index,
    `max_boost_round`) — the only sub-fit-granularity signal available, since
    `xgb.train` is a single blocking call otherwise. `on_trial` fires once a
    trial's early stopping has settled. Neither return value is used; a
    subclass observes and renders, it never influences training.
    """

    def on_round(self, epoch: int, max_round: int) -> None: ...
    def on_trial(self, trial_idx: int, n_trials: int, result: "FitResult") -> None: ...


_NULL_PROGRESS = Progress()


def _dmatrix(fm: FlatMatrix, y: np.ndarray) -> xgb.DMatrix:
    return xgb.DMatrix(fm.values, label=y, feature_types=fm.feature_types, enable_categorical=True)


def fit_one(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict,
    xgb_spec: dict,
    progress: Progress = _NULL_PROGRESS,
) -> FitResult:
    """One XGBoost fit with early stopping on `dval`."""
    s = xgb_spec["search"]
    full_params = {
        "tree_method": xgb_spec["core"]["tree_method"],
        "multi_strategy": xgb_spec["core"]["multi_strategy"],
        "eval_metric": s["eval_metric"],
        **params,
    }

    class _RoundCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            progress.on_round(epoch, s["max_boost_round"])
            return False

    booster = xgb.train(
        full_params,
        dtrain,
        num_boost_round=s["max_boost_round"],
        evals=[(dval, "val")],
        early_stopping_rounds=s["early_stopping_rounds"],
        callbacks=[_RoundCallback()],
        verbose_eval=False,
    )
    return FitResult(
        booster=booster,
        best_iteration=int(booster.best_iteration),
        best_rmse=float(booster.best_score),
        params=params,
    )


def _sample_trials(xgb_spec: dict) -> list[dict]:
    """The fixed `n_trials` draws from `kb/xgboost_spec.json`'s space, fixed seed.

    Reused identically across every (fold-or-final, set) context — the same 16
    configurations are tried everywhere, not redrawn per context, so a
    fold-to-fold difference in the winner reflects the data, not resampling.
    """
    s = xgb_spec["search"]
    rng = np.random.default_rng(s["seed"])
    space = s["space"]
    keys = list(space)
    return [{k: rng.choice(space[k]).item() for k in keys} for _ in range(s["n_trials"])]


def random_search(
    dtrain_inner: xgb.DMatrix,
    dval_inner: xgb.DMatrix,
    xgb_spec: dict,
    progress: Progress = _NULL_PROGRESS,
) -> FitResult:
    """Fit every trial in the fixed space, return the one with the best inner-val rmse."""
    trials = _sample_trials(xgb_spec)
    results = []
    for i, params in enumerate(trials):
        r = fit_one(dtrain_inner, dval_inner, params, xgb_spec, progress)
        results.append(r)
        progress.on_trial(i, len(trials), r)
    return min(results, key=lambda r: r.best_rmse)


# --------------------------------------------------------------------------
# per-context run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    context: str  # e.g. "fold_1" or "final"
    set_name: str  # "set1" or "set2"
    predictions: pd.DataFrame  # origin index, y_pred_1..y_pred_10
    winning_params: dict
    best_iteration: int
    inner_val_rmse: float
    n_train: int
    n_predict: int
    seconds: float


def run_context(
    cal: TradingCalendar,
    spec: encoding.EncodingSpec,
    globals_: encoding.GlobalEncoders,
    xgb_spec: dict,
    *,
    context: str,
    set_name: str,
    train_origins: pd.DatetimeIndex,
    inner_train_origins: pd.DatetimeIndex,
    inner_val_origins: pd.DatetimeIndex,
    predict_origins: pd.DatetimeIndex,
    set1_matrix: FlatMatrix | None,
    fold_for_pca: Any,
    progress: Progress = _NULL_PROGRESS,
) -> RunResult:
    """Search on the inner split, refit on the whole block, predict on `predict_origins`.

    `set1_matrix` is the once-built, fold-invariant set-1 matrix to slice from;
    pass `None` when `set_name == "set2"`. `fold_for_pca` is only used for set 2
    (duck-typed for `set2.fit_fold`); ignored for set 1.
    """
    start = time.monotonic()

    if set_name == "set1":
        assert set1_matrix is not None
        fm_inner_train = slice_matrix(set1_matrix, inner_train_origins)
        fm_inner_val = slice_matrix(set1_matrix, inner_val_origins)
        fm_train = slice_matrix(set1_matrix, train_origins)
        fm_predict = slice_matrix(set1_matrix, predict_origins)
    else:
        fm_inner_train = build_set2_matrix_for_fold(cal, spec, globals_, fold_for_pca, inner_train_origins)
        fm_inner_val = build_set2_matrix_for_fold(cal, spec, globals_, fold_for_pca, inner_val_origins)
        fm_train = build_set2_matrix_for_fold(cal, spec, globals_, fold_for_pca, train_origins)
        fm_predict = build_set2_matrix_for_fold(cal, spec, globals_, fold_for_pca, predict_origins)

    y_inner_train = _target_matrix(cal, inner_train_origins)
    y_inner_val = _target_matrix(cal, inner_val_origins)
    y_train = _target_matrix(cal, train_origins)

    dtrain_inner = _dmatrix(fm_inner_train, y_inner_train)
    dval_inner = _dmatrix(fm_inner_val, y_inner_val)
    search = random_search(dtrain_inner, dval_inner, xgb_spec, progress)

    full_params = {
        "tree_method": xgb_spec["core"]["tree_method"],
        "multi_strategy": xgb_spec["core"]["multi_strategy"],
        **search.params,
    }
    dtrain_full = _dmatrix(fm_train, y_train)
    refit_rounds = search.best_iteration + 1

    class _RefitCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            progress.on_round(epoch, refit_rounds)
            return False

    final_booster = xgb.train(
        full_params, dtrain_full, num_boost_round=refit_rounds, callbacks=[_RefitCallback()]
    )

    dpredict = xgb.DMatrix(
        fm_predict.values, feature_types=fm_predict.feature_types, enable_categorical=True
    )
    preds = final_booster.predict(dpredict)

    predictions = pd.DataFrame(
        preds, index=predict_origins, columns=[f"y_pred_{k}" for k in range(1, preds.shape[1] + 1)]
    )
    predictions.index.name = "origin"

    return RunResult(
        context=context,
        set_name=set_name,
        predictions=predictions,
        winning_params=search.params,
        best_iteration=search.best_iteration,
        inner_val_rmse=search.best_rmse,
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

    fm1 = build_set1_matrix(cal, spec, globals_, fold1.train)
    print(f"set1 matrix: {fm1.values.shape}  finite={np.isfinite(fm1.values).all()}")
    assert fm1.values.shape[1] == 10 * 90 + 10, "set1 flat width should be 10 cols x 90 + 10 elapsed"

    fm2 = build_set2_matrix_for_fold(cal, spec, globals_, fold1, fold1.train)
    print(f"set2 matrix: {fm2.values.shape}  finite={np.isfinite(fm2.values).all()}")
    assert fm2.values.shape[1] == 49 * 90 + 10, "set2 flat width should be 49 cols x 90 + 10 elapsed"

    xgb_spec = load_xgboost_spec()
    print(f"xgboost spec: {xgb_spec['core']['multi_strategy']}, "
          f"{xgb_spec['search']['n_trials']} trials")


if __name__ == "__main__":
    main()
