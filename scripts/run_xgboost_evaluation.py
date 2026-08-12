"""Score the real XGBoost predictions (#38) through #14's locked evaluation protocol.

Reads `predictions/xgboost/*.parquet`, builds the position/cost/scorecard
pipeline `prdict/evaluation.py` defines, and reports Sharpe/cum-return/max-DD/
hit-rate per fold and aggregate for set 1, set 2, and the always-up baseline.

Alignment note: `evaluation.py`'s `daily_return` must be the *forward* realized
return that lands the session after each origin (an origin's own close is
already known when its prediction is formed, so accruing that day's return
would be look-ahead). That series is exactly `targets.build(cal).y[:, 0]`
indexed by origin -- reused here rather than re-derived.

    uv run python scripts/run_xgboost_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from prdict import evaluation as ev
from prdict.folds import load_fold_spec
from prdict.targets import build as build_target, step_returns
from prdict.trading_calendar import load_calendar

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "predictions" / "xgboost"
OUT_MD = ROOT / "kb" / "xgboost_evaluation.md"

CONTEXTS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "final"]
SETS = ["set1", "set2"]


def _forward_return(cal, origins: pd.DatetimeIndex) -> pd.Series:
    """The realized next-session return for each origin -- y_1 of the scored target."""
    tgt = build_target(cal)
    row = {o: i for i, o in enumerate(tgt.origins)}
    missing = [o for o in origins if o not in row]
    if missing:
        raise ValueError(f"{len(missing)} origins have no scored target, e.g. {missing[0].date()}")
    y1 = tgt.y[[row[o] for o in origins], 0]
    return pd.Series(y1, index=origins)


def _trailing_vol(cal, spec: ev.EvaluationSpec, origins: pd.DatetimeIndex) -> pd.Series:
    """Trailing realized vol as of each origin's own close -- backward-looking only."""
    daily = step_returns(cal)
    vol = ev.realized_vol(daily, spec.vol_window, spec.horizon)
    missing = [o for o in origins if o not in vol.index]
    if missing:
        raise ValueError(f"{len(missing)} origins missing from the vol series, e.g. {missing[0].date()}")
    return vol.loc[origins]


def _score_one(cal, spec: ev.EvaluationSpec, pred: pd.DataFrame) -> tuple[ev.Scorecard, ev.Scorecard]:
    """(model_scorecard, baseline_scorecard) for one context/set's predictions."""
    origins = pred.index
    cum_pred = pred.sum(axis=1)
    vol = _trailing_vol(cal, spec, origins)
    raw = ev.raw_signal(cum_pred, vol)
    sized = ev.target_size(raw, spec.neutral_zone)

    daily_return = _forward_return(cal, origins)
    trades = ev.simulate_single_position(sized)
    model_returns = ev.strategy_returns(trades, sized, daily_return, spec.entry_bps, spec.exit_bps)
    base_returns = ev.baseline_returns(daily_return, spec.baseline_entry_bps, spec.exit_bps)

    actual_dir = np.sign(daily_return)
    model_sc = ev.scorecard(model_returns, sized.apply(np.sign), actual_dir, spec.sessions_per_year)
    base_sc = ev.scorecard(base_returns, pd.Series(1, index=daily_return.index), actual_dir, spec.sessions_per_year)
    return model_sc, base_sc


def _fmt(sc: ev.Scorecard) -> str:
    hr = f"{sc.hit_rate:.3f}" if not np.isnan(sc.hit_rate) else "n/a"
    return f"sharpe={sc.sharpe:+.3f}  cum_ret={sc.cumulative_return:+.4f}  max_dd={sc.max_drawdown:+.4f}  hit_rate={hr}"


def main() -> None:
    cal = load_calendar()
    spec = ev.load_spec()

    print(f"Evaluation protocol: neutral_zone={spec.neutral_zone}  vol_window={spec.vol_window}  "
          f"entry_bps={spec.entry_bps}  exit_bps={spec.exit_bps}  baseline_bps={spec.baseline_entry_bps}\n")

    rows = []
    agg_returns = {s: [] for s in SETS}
    agg_base_returns = []
    agg_pos_dir = {s: [] for s in SETS}
    agg_base_pos_dir = []
    agg_actual_dir = []

    for ctx in CONTEXTS:
        base_sc_for_ctx = None
        for set_name in SETS:
            path = PRED_DIR / f"{ctx}_{set_name}.parquet"
            pred = pd.read_parquet(path)
            model_sc, base_sc = _score_one(cal, spec, pred)
            base_sc_for_ctx = base_sc

            print(f"{ctx:8s} {set_name}  model:    {_fmt(model_sc)}")
            rows.append((ctx, set_name, model_sc))

            origins = pred.index
            cum_pred = pred.sum(axis=1)
            vol = _trailing_vol(cal, spec, origins)
            sized = ev.target_size(ev.raw_signal(cum_pred, vol), spec.neutral_zone)
            daily_return = _forward_return(cal, origins)
            trades = ev.simulate_single_position(sized)
            model_returns = ev.strategy_returns(trades, sized, daily_return, spec.entry_bps, spec.exit_bps)

            if ctx != "final":  # aggregate over the 5 rolling folds only, holdout kept separate
                agg_returns[set_name].append(model_returns)
                agg_pos_dir[set_name].append(sized.apply(np.sign))
                if set_name == "set1":
                    agg_actual_dir.append(np.sign(daily_return))

        print(f"{ctx:8s} baseline  {_fmt(base_sc_for_ctx)}")
        if ctx != "final":
            path = PRED_DIR / f"{ctx}_set1.parquet"
            origins = pd.read_parquet(path).index
            daily_return = _forward_return(cal, origins)
            base_returns = ev.baseline_returns(daily_return, spec.baseline_entry_bps, spec.exit_bps)
            agg_base_returns.append(base_returns)
            agg_base_pos_dir.append(pd.Series(1, index=daily_return.index))
        print()

    print("=== Aggregate across the 5 rolling folds (holdout reported separately above) ===")
    actual_dir_all = pd.concat(agg_actual_dir)
    for set_name in SETS:
        returns_all = pd.concat(agg_returns[set_name])
        pos_dir_all = pd.concat(agg_pos_dir[set_name])
        sc = ev.scorecard(returns_all, pos_dir_all, actual_dir_all, spec.sessions_per_year)
        print(f"{set_name}      model:    {_fmt(sc)}")
        rows.append(("aggregate", set_name, sc))

    base_returns_all = pd.concat(agg_base_returns)
    base_pos_dir_all = pd.concat(agg_base_pos_dir)
    base_sc = ev.scorecard(base_returns_all, base_pos_dir_all, actual_dir_all, spec.sessions_per_year)
    print(f"baseline  {_fmt(base_sc)}")
    rows.append(("aggregate", "baseline", base_sc))

    lines = [
        "# XGBoost evaluation (#38 x #14)",
        "",
        "Generated by `scripts/run_xgboost_evaluation.py` against the real predictions in "
        "`predictions/xgboost/*.parquet`, scored through the locked protocol in `kb/evaluation_spec.md`.",
        "",
        "| Context | Set | Sharpe | Cum return | Max DD | Hit rate |",
        "|---|---|---|---|---|---|",
    ]
    for ctx, set_name, sc in rows:
        hr = f"{sc.hit_rate:.3f}" if not np.isnan(sc.hit_rate) else "n/a"
        lines.append(f"| {ctx} | {set_name} | {sc.sharpe:+.3f} | {sc.cumulative_return:+.4f} | {sc.max_drawdown:+.4f} | {hr} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWritten: {OUT_MD}")


if __name__ == "__main__":
    main()
