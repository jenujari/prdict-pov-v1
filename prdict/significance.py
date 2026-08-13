"""Target-permutation ("y-scrambling") significance test for #40's flagged
holdout result (#54).

Contract fixed in `kb/significance_spec.json` (`scripts/build_significance_spec.py`)
before any permutation ran — this module only ever reads that file. Reuses
#38's real, already-fitted pipeline pieces unchanged: `xgboost_model`'s set2
matrix builder and `DMatrix` helper, `evaluation.py`'s unmodified scoring
chain. The only thing this module adds is the permutation itself.

One `PermutationContext` (the fold-invariant set2 PCA matrix for
`final_train` + `holdout`, and the real target array) is built **once** and
reused for every permutation trial — the PCA fit depends only on features,
never on labels, so it doesn't need refitting per shuffle.

    uv run python -m prdict.significance    # self-check: real result reproduces exactly
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb

from prdict import encoding
from prdict import evaluation as ev
from prdict.dataset import FlatMatrix
from prdict.folds import load_fold_spec
from prdict.targets import build as build_target
from prdict.targets import step_returns
from prdict.trading_calendar import TradingCalendar, load_calendar
from prdict.xgboost_model import _dmatrix, _target_matrix, build_set2_matrix_for_fold, load_xgboost_spec, slice_matrix

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "kb" / "significance_spec.json"


def load_significance_spec(path: Path = SPEC_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/build_significance_spec.py first.")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# fold-invariant context — built once, reused by every permutation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationContext:
    fm_train: FlatMatrix
    fm_holdout: FlatMatrix
    y_train: np.ndarray  # real (n_train, 10) target -- shuffled per-permutation, never mutated here
    train_origins: pd.DatetimeIndex
    holdout_origins: pd.DatetimeIndex


def build_context(
    cal: TradingCalendar, spec: encoding.EncodingSpec, globals_: encoding.GlobalEncoders, fs
) -> PermutationContext:
    final_train = fs.final_train(cal)
    holdout = fs.holdout(cal)
    final_block = SimpleNamespace(train=final_train)

    all_origins = final_train.append(holdout).unique().sort_values()
    fm_all = build_set2_matrix_for_fold(cal, spec, globals_, final_block, all_origins)
    fm_train = slice_matrix(fm_all, final_train)
    fm_holdout = slice_matrix(fm_all, holdout)
    y_train = _target_matrix(cal, final_train)

    return PermutationContext(
        fm_train=fm_train, fm_holdout=fm_holdout, y_train=y_train,
        train_origins=final_train, holdout_origins=holdout,
    )


# --------------------------------------------------------------------------
# scoring — evaluation.py's chain, unchanged, on an in-memory prediction frame
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringContext:
    """Everything `score_predictions` needs that doesn't depend on the model —
    built once, shared across every permutation and the real-result self-check."""

    tgt: object
    trailing_vol: pd.Series
    eval_spec: ev.EvaluationSpec


def build_scoring_context(cal: TradingCalendar) -> ScoringContext:
    eval_spec = ev.load_spec()
    tgt = build_target(cal)
    trailing_vol = ev.realized_vol(step_returns(cal), eval_spec.vol_window, eval_spec.horizon)
    return ScoringContext(tgt=tgt, trailing_vol=trailing_vol, eval_spec=eval_spec)


def score_predictions(predictions: pd.DataFrame, sc: ScoringContext) -> ev.Scorecard:
    """#14's unchanged position/cost/scorecard chain on an in-memory prediction frame."""
    origins = predictions.index
    cum_return_pred = predictions.sum(axis=1)
    vol = sc.trailing_vol.loc[origins]
    row = {o: i for i, o in enumerate(sc.tgt.origins)}
    daily_return = pd.Series(sc.tgt.y[[row[o] for o in origins], 0], index=origins)

    raw = ev.raw_signal(cum_return_pred, vol)
    sized = ev.target_size(raw, sc.eval_spec.neutral_zone)
    trades = ev.simulate_single_position(sized)
    returns = ev.strategy_returns(trades, sized, daily_return, sc.eval_spec.entry_bps, sc.eval_spec.exit_bps)
    position_direction = sized.apply(np.sign)
    actual_direction = np.sign(daily_return)
    return ev.scorecard(returns, position_direction, actual_direction, sc.eval_spec.sessions_per_year)


# --------------------------------------------------------------------------
# one permutation trial
# --------------------------------------------------------------------------


def fit_and_predict(ctx: PermutationContext, xgb_core: dict, sig_spec: dict, y_train: np.ndarray) -> pd.DataFrame:
    """One XGBoost fit at the real winning config + fixed round count, predicting
    on the real holdout. `y_train` is the caller's choice — the real target for
    the self-check, a shuffled one for a permutation trial."""
    full_params = {
        "tree_method": xgb_core["tree_method"],
        "multi_strategy": xgb_core["multi_strategy"],
        **sig_spec["winning_params"],
    }
    dtrain = _dmatrix(ctx.fm_train, y_train)
    booster = xgb.train(full_params, dtrain, num_boost_round=sig_spec["num_boost_round"])

    dpredict = xgb.DMatrix(
        ctx.fm_holdout.values, feature_types=ctx.fm_holdout.feature_types, enable_categorical=True
    )
    preds = booster.predict(dpredict)
    predictions = pd.DataFrame(
        preds, index=ctx.holdout_origins, columns=[f"y_pred_{k}" for k in range(1, preds.shape[1] + 1)]
    )
    predictions.index.name = "origin"
    return predictions


def run_permutation(
    ctx: PermutationContext, xgb_core: dict, sig_spec: dict, sc: ScoringContext, seed: int
) -> float:
    """One null trial: shuffle final_train's target rows, refit, predict on the
    real holdout, score against real holdout returns. Returns the null Sharpe."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ctx.train_origins))
    y_shuffled = ctx.y_train[perm]

    predictions = fit_and_predict(ctx, xgb_core, sig_spec, y_shuffled)
    card = score_predictions(predictions, sc)
    return card.sharpe


# --------------------------------------------------------------------------
# secondary test — binomial hit-rate vs 0.5, per model/set arm
# --------------------------------------------------------------------------


PRED_DIRS = {"xgboost": ROOT / "predictions" / "xgboost", "tft": ROOT / "predictions" / "tft"}
FOLD_CONTEXTS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]


def _position_and_actual(pred_path: Path, sc: ScoringContext) -> tuple[pd.Series, pd.Series]:
    """Real, already-committed predictions -> (position_direction, actual_direction)."""
    pred = pd.read_parquet(pred_path)
    origins = pred.index
    cum_return_pred = pred.sum(axis=1)
    vol = sc.trailing_vol.loc[origins]
    row = {o: i for i, o in enumerate(sc.tgt.origins)}
    daily_return = pd.Series(sc.tgt.y[[row[o] for o in origins], 0], index=origins)

    raw = ev.raw_signal(cum_return_pred, vol)
    sized = ev.target_size(raw, sc.eval_spec.neutral_zone)
    return sized.apply(np.sign), np.sign(daily_return)


# kb/fold_spec.md §6: "Consecutive origins share 9 of 10 label days... this
# governs interpretation: confidence intervals and any significance test must
# use the effective count, not the origin count." A run of 10 consecutive
# overlapping-label origins is worth roughly 1 independent observation.
EFFECTIVE_SAMPLE_RATIO = 1 / 10


def hit_rate_binomial_tests(sc: ScoringContext) -> list[dict]:
    """Binomial test of each model/set arm's directional hit-rate against 0.5 —
    reuses the real, already-committed predictions from #38/#39 (no retraining).
    One row per (model, set, view), view in {aggregate (5 folds pooled), holdout}.

    Run at the EFFECTIVE sample size (kb/fold_spec.md §6's ~1/10 discount for
    9-of-10 label overlap), not the raw origin count -- the raw count is
    reported alongside for transparency, but treating ~2700 overlapping daily
    calls as 2700 independent Bernoulli trials would be exactly the mistake
    the fold spec warns against, and would make a real-but-small edge look far
    more significant than the data actually supports.
    """
    from scipy.stats import binomtest

    rows: list[dict] = []
    for model in ("xgboost", "tft"):
        for set_name in ("set1", "set2"):
            agg_pos, agg_actual = [], []
            for context in FOLD_CONTEXTS:
                pos, actual = _position_and_actual(PRED_DIRS[model] / f"{context}_{set_name}.parquet", sc)
                agg_pos.append(pos)
                agg_actual.append(actual)
            views = {
                "aggregate": (pd.concat(agg_pos), pd.concat(agg_actual)),
                "holdout": _position_and_actual(PRED_DIRS[model] / f"final_{set_name}.parquet", sc),
            }
            for view_name, (pos, actual) in views.items():
                mask = (pos != 0) & (actual != 0)
                n_calls_raw = int(mask.sum())
                n_hits_raw = int((pos[mask] == actual[mask]).sum())
                hit_rate = n_hits_raw / n_calls_raw if n_calls_raw else float("nan")

                n_calls_eff = max(round(n_calls_raw * EFFECTIVE_SAMPLE_RATIO), 1)
                n_hits_eff = round(hit_rate * n_calls_eff) if n_calls_raw else 0
                result = binomtest(n_hits_eff, n_calls_eff, p=0.5, alternative="two-sided")
                rows.append(
                    {
                        "model": model,
                        "set": set_name,
                        "view": view_name,
                        "n_calls_raw": n_calls_raw,
                        "n_hits_raw": n_hits_raw,
                        "hit_rate": hit_rate,
                        "n_calls_effective": n_calls_eff,
                        "p_value": result.pvalue,
                    }
                )
    return rows


def main() -> None:
    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)
    fs = load_fold_spec()
    xgb_spec = load_xgboost_spec()
    sig_spec = load_significance_spec()

    ctx = build_context(cal, spec, globals_, fs)
    sc = build_scoring_context(cal)
    print(f"context built: fm_train {ctx.fm_train.values.shape}  fm_holdout {ctx.fm_holdout.values.shape}")

    # Self-check: the REAL (unshuffled) target, through this exact code path,
    # must reproduce #40's committed holdout Sharpe -- proof the harness scores
    # what it claims to, not a silent bug that would invalidate every permutation.
    real_predictions = fit_and_predict(ctx, xgb_spec["core"], sig_spec, ctx.y_train)
    real_card = score_predictions(real_predictions, sc)
    observed = sig_spec["target"]["observed_value"]
    print(f"real (unshuffled) holdout sharpe: {real_card.sharpe:.4f}  (spec's recorded observed: {observed})")
    assert abs(real_card.sharpe - observed) < 0.01, (
        f"self-check failed: real result {real_card.sharpe:.4f} doesn't reproduce "
        f"the spec's recorded {observed} -- the scoring path is inconsistent with #40's"
    )
    print("self-check passed: real result reproduces exactly through this code path")

    # One real permutation trial, to confirm the null construction actually runs.
    null_sharpe = run_permutation(ctx, xgb_spec["core"], sig_spec, sc, seed=sig_spec["budget"]["seed_base"])
    print(f"one null trial (seed={sig_spec['budget']['seed_base']}): sharpe={null_sharpe:.4f}")

    print("\nsecondary: binomial hit-rate tests, at the EFFECTIVE sample size (kb/fold_spec.md §6)")
    for row in hit_rate_binomial_tests(sc):
        print(
            f"  {row['model']}/{row['set']}/{row['view']}: hit_rate={row['hit_rate']:.3f} "
            f"(n_raw={row['n_calls_raw']}, n_effective={row['n_calls_effective']})  p={row['p_value']:.4f}"
        )


if __name__ == "__main__":
    main()
