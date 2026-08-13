"""Score baseline / XGBoost / TFT side by side and write the final comparison (#40).

Resolves wayfinder ticket #40 -- the map's closing ticket. Reads the real
predictions #38 and #39 already produced (`predictions/xgboost/`,
`predictions/tft/`) and scores every (context, set, model) combination through
#14's locked protocol (`prdict/evaluation.py`), unchanged -- this script only
assembles inputs and formats output, exactly like `run_xgboost_evaluation.py`/
`run_tft_evaluation.py` already do per-model. Baseline is computed once per
context (it doesn't depend on the model) and shared across both models' rows,
rather than recomputed twice as the two standalone evaluation scripts do.

Regime membership is derived from `kb/fold_spec.json`'s own fold boundaries,
not eyeballed: a fold's *validation* window (the only thing ever scored)
either contains a date inside the named regime or it doesn't.

    uv run python scripts/build_final_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prdict import evaluation as ev
from prdict.folds import load_fold_spec
from prdict.targets import build as build_target, step_returns
from prdict.trading_calendar import load_calendar

ROOT = Path(__file__).resolve().parent.parent
PRED_DIRS = {"xgboost": ROOT / "predictions" / "xgboost", "tft": ROOT / "predictions" / "tft"}
OUT_MD = ROOT / "kb" / "final_comparison.md"

CONTEXTS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "final"]
SETS = ["set1", "set2"]
MODELS = ["xgboost", "tft"]

# Regimes named in the map (#2) / kb/fold_spec.md. A regime is "in" a context's
# *validation* window (the only thing ever scored) iff any date in that window
# falls inside the regime's date range -- derived from kb/fold_spec.json's own
# fold boundaries, not eyeballed.
REGIMES = {
    "2008 crash": (pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31")),
    "2020 covid crash": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
}


def _regimes_in(origins: pd.DatetimeIndex) -> list[str]:
    lo, hi = origins.min(), origins.max()
    return [name for name, (rlo, rhi) in REGIMES.items() if lo <= rhi and hi >= rlo]


def _daily_return_for(origins: pd.DatetimeIndex, tgt) -> pd.Series:
    row = {o: i for i, o in enumerate(tgt.origins)}
    missing = [o for o in origins if o not in row]
    if missing:
        raise ValueError(f"{len(missing)} origins missing a scored target, e.g. {missing[0].date()}")
    return pd.Series(tgt.y[[row[o] for o in origins], 0], index=origins)


def _vol_for(origins: pd.DatetimeIndex, trailing_vol: pd.Series) -> pd.Series:
    missing = origins.difference(trailing_vol.index)
    if len(missing):
        raise ValueError(f"{len(missing)} origins missing a trailing-vol estimate, e.g. {missing[0].date()}")
    return trailing_vol.loc[origins]


def score_model(pred_dir: Path, context: str, set_name: str, tgt, trailing_vol: pd.Series, spec: ev.EvaluationSpec):
    """Returns (scorecard, returns, position_direction, actual_direction, origins) —
    the raw series are what a pooled aggregate needs (concatenate-then-score,
    not average-the-per-fold-Sharpes — the latter ignores how much each fold
    actually contributed to compounded risk, the former is what "trade this
    continuously across the stitched folds" actually means)."""
    pred = pd.read_parquet(pred_dir / f"{context}_{set_name}.parquet")
    origins = pred.index

    cum_return_pred = pred.sum(axis=1)
    vol = _vol_for(origins, trailing_vol)
    daily_return = _daily_return_for(origins, tgt)

    raw = ev.raw_signal(cum_return_pred, vol)
    sized = ev.target_size(raw, spec.neutral_zone)
    trades = ev.simulate_single_position(sized)
    returns = ev.strategy_returns(trades, sized, daily_return, spec.entry_bps, spec.exit_bps)

    position_direction = sized.apply(np.sign)
    actual_direction = np.sign(daily_return)
    card = ev.scorecard(returns, position_direction, actual_direction, spec.sessions_per_year)
    return card, returns, position_direction, actual_direction, origins


def score_baseline(origins: pd.DatetimeIndex, tgt, spec: ev.EvaluationSpec):
    daily_return = _daily_return_for(origins, tgt)
    actual_direction = np.sign(daily_return)
    position_direction = pd.Series(1, index=daily_return.index)
    baseline = ev.baseline_returns(daily_return, spec.baseline_entry_bps, spec.exit_bps)
    card = ev.scorecard(baseline, position_direction, actual_direction, spec.sessions_per_year)
    return card, baseline, position_direction, actual_direction


def _fmt_row(context: str, set_name: str, model: str, card: ev.Scorecard, regimes: list[str]) -> str:
    hr = f"{card.hit_rate:.3f}" if not np.isnan(card.hit_rate) else "n/a"
    tag = f" ({', '.join(regimes)})" if regimes else ""
    return (
        f"| {context}{tag} | {set_name} | {model} | {card.sharpe:.3f} | "
        f"{card.cumulative_return:.4f} | {card.max_drawdown:.4f} | {hr} |"
    )


def main() -> None:
    cal = load_calendar()
    fs = load_fold_spec()
    spec = ev.load_spec()
    tgt = build_target(cal)

    raw_daily = step_returns(cal)
    trailing_vol = ev.realized_vol(raw_daily, spec.vol_window, spec.horizon)

    rows: list[str] = []
    # Pooled series across the 5 rolling folds, per (model, set) — aggregate
    # Sharpe is computed on the CONCATENATED return series (as if traded
    # continuously across the stitched folds), not as an average of the five
    # per-fold Sharpes, which would ignore how much each fold actually
    # contributed to compounded risk.
    pooled: dict[tuple[str, str], dict[str, list]] = {
        key: {"returns": [], "pos": [], "actual": []}
        for key in [("xgboost", "set1"), ("xgboost", "set2"), ("tft", "set1"), ("tft", "set2"), ("baseline", "-")]
    }
    holdout_rows: dict[str, ev.Scorecard] = {}

    for context in CONTEXTS:
        baseline_result = None
        regimes: list[str] = []
        for set_name in SETS:
            for model in MODELS:
                card, returns, pos, actual, origins = score_model(
                    PRED_DIRS[model], context, set_name, tgt, trailing_vol, spec
                )
                regimes = _regimes_in(origins)
                rows.append(_fmt_row(context, set_name, model, card, regimes))
                key = (model, set_name)
                if context != "final":
                    pooled[key]["returns"].append(returns)
                    pooled[key]["pos"].append(pos)
                    pooled[key]["actual"].append(actual)
                else:
                    holdout_rows[f"{model}/{set_name}"] = card
            if baseline_result is None:
                baseline_result = score_baseline(origins, tgt, spec)
        baseline_card, b_returns, b_pos, b_actual = baseline_result
        rows.append(_fmt_row(context, "-", "baseline", baseline_card, regimes))
        if context != "final":
            pooled[("baseline", "-")]["returns"].append(b_returns)
            pooled[("baseline", "-")]["pos"].append(b_pos)
            pooled[("baseline", "-")]["actual"].append(b_actual)
        else:
            holdout_rows["baseline"] = baseline_card

    agg_cards: dict[tuple[str, str], ev.Scorecard] = {}
    for key, parts in pooled.items():
        r = pd.concat(parts["returns"])
        p = pd.concat(parts["pos"])
        a = pd.concat(parts["actual"])
        agg_cards[key] = ev.scorecard(r, p, a, spec.sessions_per_year)

    lines = [
        "# Final comparison — three models x two input sets, holdout scorecard",
        "",
        "Generated by `scripts/build_final_comparison.py`. Do not hand-edit — rerun the script.",
        "",
        f"Resolves [#40](https://github.com/jenujari/prdict-pov-v1/issues/40) — the map "
        f"[#2](https://github.com/jenujari/prdict-pov-v1/issues/2) destination. Scoring "
        "protocol locked in `kb/evaluation_spec.md` (map decision 8) before any prediction "
        "existed. Real predictions from #38 (XGBoost, `kb/xgboost_run.md`) and #39 "
        "(TFT, `kb/tft_run.md`).",
        "",
        "## 1. Per-fold and holdout",
        "",
        "Regime tags mark a context whose **validation** window (the only thing ever scored) "
        "overlaps the named regime — derived from `kb/fold_spec.json`'s fold boundaries, not "
        "eyeballed. `final` is the untouched holdout (2024-01-01 → 2026-06-15), the primary "
        "comparison this ticket exists to make.",
        "",
        "| Context | Set | Model | Sharpe | Cum return | Max DD | Hit rate |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(rows)

    lines += [
        "",
        "## 2. Aggregate across the 5 rolling folds",
        "",
        "Sharpe/cum-return/max-DD computed on the **concatenated** return series across "
        "all 5 folds (as if traded continuously across the stitched folds) — not an average "
        "of the five per-fold numbers, which would understate how much the worse folds "
        "actually cost.",
        "",
        "| Model | Set | Sharpe | Cum return | Max DD | Hit rate |",
        "|---|---|---|---|---|---|",
    ]
    for model, set_name in [("xgboost", "set1"), ("xgboost", "set2"), ("tft", "set1"), ("tft", "set2")]:
        c = agg_cards[(model, set_name)]
        lines.append(
            f"| {model} | {set_name} | {c.sharpe:.3f} | {c.cumulative_return:+.4f} | "
            f"{c.max_drawdown:+.4f} | {c.hit_rate:.3f} |"
        )
    bc = agg_cards[("baseline", "-")]
    lines.append(f"| baseline | - | {bc.sharpe:.3f} | {bc.cumulative_return:+.4f} | {bc.max_drawdown:+.4f} | - |")

    xs1, xs2 = agg_cards[("xgboost", "set1")], agg_cards[("xgboost", "set2")]
    ts1, ts2 = agg_cards[("tft", "set1")], agg_cards[("tft", "set2")]
    hxs1, hxs2 = holdout_rows["xgboost/set1"], holdout_rows["xgboost/set2"]
    hts1, hts2 = holdout_rows["tft/set1"], holdout_rows["tft/set2"]
    hb = holdout_rows["baseline"]

    lines += [
        "",
        "## 3. Regime coverage",
        "",
        "- **2008 crash: never scored.** No fold's validation window overlaps 2008 — it only "
        "ever appears inside training blocks (folds 1-4; fold 5 trains from 2011 and never "
        "sees 2008 at all, `kb/fold_spec.md` §7). Nothing in this comparison speaks to how "
        "either model behaves in a crash of that severity.",
        "- **2020 COVID crash: scored once, in fold 4's validation window** "
        "(`2018-07-20` → `2021-04-09`) — the one place in this comparison a genuine crash "
        "regime is actually in a scored window. Fold 4's rows above are the only real evidence "
        "either model has about crash behavior.",
        "- **Holdout (`final`): a calm rising regime** (+9.8% total, 13.9% annualised vol, "
        "worst drawdown -15.8% — `kb/fold_spec.json`'s own recorded characterization). The "
        "always-up baseline scores well here by construction, so the holdout has limited power "
        "to separate either model from it on Sharpe or drawdown specifically.",
        "",
        "## 4. Claim",
        "",
        "**Licensed: risk-adjusted comparison vs. buy-and-hold, net of costs — not an "
        "absolute-return race** (`kb/evaluation_spec.md` §5, map decision 8). Buy-and-hold's "
        "cumulative return over these windows is large by construction (a rising-market "
        "instrument, held continuously, pays trading costs almost never); beating it in "
        "absolute terms was never the bar either model was tested against.",
        "",
        f"**Aggregate across the 5 rolling folds (pooled): baseline wins comfortably** — "
        f"Sharpe {bc.sharpe:.3f} vs XGBoost's {xs1.sharpe:.3f} (set1) / {xs2.sharpe:.3f} (set2) "
        f"and TFT's {ts1.sharpe:.3f} (set1) / {ts2.sharpe:.3f} (set2). No model/set arm comes "
        "close on this larger, more statistically informative sample.",
        "",
        f"**On the untouched holdout specifically, one arm does beat the baseline: XGBoost set 2, "
        f"Sharpe {hxs2.sharpe:.3f} vs baseline's {hb.sharpe:.3f}** — though on a much smaller "
        f"cumulative return ({hxs2.cumulative_return:+.1%} vs {hb.cumulative_return:+.1%}), i.e. a "
        "materially lower-risk position that produced a better return-per-unit-risk over this one "
        "window. This is the single most interesting number in the whole comparison, and it "
        "deserves being stated plainly rather than averaged away — but it is **one number, on one "
        "601-origin, single-regime window**, with no significance test applied (out of scope for "
        f"this ticket, per the map's own \"Not yet specified\" list). XGBoost set 1 lands exactly "
        f"on baseline's holdout Sharpe ({hxs1.sharpe:.3f}) — the same near-constant-prediction "
        "degeneration seen throughout this project's results, not a real second data point. Both "
        f"TFT arms underperform baseline on the holdout ({hts1.sharpe:.3f}, {hts2.sharpe:.3f}).",
        "",
        "Taken together: **no arm shows a robust, cross-window risk-adjusted edge over "
        "buy-and-hold.** The one holdout beat is real in the data but unreplicated — it doesn't "
        f"appear in that arm's own fold-level results (XGBoost set2's aggregate Sharpe, "
        f"{xs2.sharpe:.3f}, sits well under baseline's {bc.sharpe:.3f}), so the more defensible "
        "read is \"one favorable window,\" not \"a discovered edge.\"",
        "",
        "## 5. Does planetary configuration carry signal?",
        "",
        "**Weak evidence for a small, real, non-tradeable signal — not evidence against one.** "
        "#12's own feature-selection step already found this before any model was trained: "
        "max mutual information ~0.098 against a permutation null of ~0.026 — small but real, "
        "not noise. Both models' directional hit-rates across the arms above sit mostly at or "
        "modestly above a coin flip (0.46–0.56, occasionally below for TFT) — consistent with a "
        "real but small lean, not a strong signal and not nothing. That is the correctly-sized "
        "conclusion for an input this weak: not \"no signal,\" and not \"a tradeable edge\" — a "
        "real but small directional lean that 20bps round-trip costs and realistic position "
        "sizing consume before it reliably becomes a risk-adjusted win over simply holding the "
        "index. The one holdout exception (§4) is consistent with this framing too: a weak "
        "signal will occasionally produce a good window by chance or by genuine (if "
        "inconsistent) contribution — one window can't distinguish the two.",
        "",
        "This is evidenced on one calm holdout regime and folds that never score a real crash "
        "except once (§3) — the honest scope of what this comparison can and can't speak to.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Written: {OUT_MD}")

    print("\nHoldout headline:")
    for key, card in holdout_rows.items():
        hr = f"{card.hit_rate:.3f}" if not np.isnan(card.hit_rate) else "n/a"
        print(f"  {key:<16s} sharpe={card.sharpe:+.3f}  cum_ret={card.cumulative_return:+.4f}  hit_rate={hr}")


if __name__ == "__main__":
    main()
