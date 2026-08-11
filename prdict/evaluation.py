"""Trading-simulation scorecard: position rule, cost model, metrics, baseline.

Resolves wayfinder ticket #14 (position rule / holding period / costs / metrics
/ always-up baseline / benchmark framing / regime reporting -- issue #14 body,
points 1-7). Spec and rationale at kb/evaluation_spec.md, parameters at
kb/evaluation_spec.json.

No model has been trained yet (#38/#39 do that) -- this module is the locked
protocol those tickets score predictions against, so `main()` self-checks the
mechanics against small synthetic series rather than real predictions. Locking
the mechanism before a result exists is the point: map decision 8 requires the
scorecard be settled *before* any model result is looked at.

    uv run python -m prdict.evaluation    # self-check against synthetic series
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPEC_JSON = ROOT / "kb" / "evaluation_spec.json"


@dataclass(frozen=True)
class EvaluationSpec:
    """The locked scorecard parameters, read from kb/evaluation_spec.json."""

    spec: dict

    @property
    def neutral_zone(self) -> float:
        return self.spec["position_rule"]["sizing"]["neutral_zone_threshold"]

    @property
    def vol_window(self) -> int:
        return self.spec["position_rule"]["sizing"]["vol_window_sessions"]

    @property
    def horizon(self) -> int:
        return self.spec["reporting"]["horizon_sessions"]

    @property
    def entry_bps(self) -> float:
        return self.spec["costs"]["entry_bps"]

    @property
    def exit_bps(self) -> float:
        return self.spec["costs"]["exit_bps"]

    @property
    def baseline_entry_bps(self) -> float:
        return self.spec["baseline"]["cost_bps"]

    @property
    def sessions_per_year(self) -> int:
        return self.spec["metrics"]["trading_sessions_per_year"]


def load_spec(path: Path = SPEC_JSON) -> EvaluationSpec:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return EvaluationSpec(spec=json.loads(path.read_text()))


# --- Position rule (kb/evaluation_spec.md section 1) ------------------------


def realized_vol(daily_return: pd.Series, window: int, horizon: int) -> pd.Series:
    """Trailing realized vol, scaled to the horizon so it sits on the same
    cumulative scale as a horizon-length predicted return -- see section 1."""
    return daily_return.rolling(window).std(ddof=1) * np.sqrt(horizon)


def raw_signal(cum_return_pred: pd.Series, vol_estimate: pd.Series) -> pd.Series:
    """Predicted-return-to-realized-vol ratio, before the neutral zone and clip.

    Dimensionless and self-normalizing by construction -- no separate scale
    constant (#14 round 2 Q8; see kb/evaluation_spec.json "why_no_scale_constant").
    """
    return cum_return_pred / vol_estimate


def target_size(raw: pd.Series, neutral_zone: float) -> pd.Series:
    """Signed position size in [-1, 1], zero inside the neutral zone."""
    size = raw.clip(-1, 1)
    return size.where(raw.abs() >= neutral_zone, 0.0)


# --- Position management (kb/evaluation_spec.md section 2) ------------------


@dataclass(frozen=True)
class Trade:
    """One single-position trade. `entry_pos`/`exit_pos` are positions into the
    origin index the strategy ran over, so `exit_pos` is exclusive of the
    holding period -- the origin at `exit_pos` is the day the exit fires, not a
    day the trade was held through."""

    entry_pos: int
    exit_pos: int
    direction: int  # +1 or -1
    size: float  # fixed at entry, in (0, 1]


def simulate_single_position(sized_target: pd.Series) -> list[Trade]:
    """The single-position-at-a-time state machine.

    `sized_target` is already through `target_size` (zero means neutral). Flat
    until the signal leaves the neutral zone, then holds a fixed-size position
    (entry-day size, never resized) until the signal's sign flips or it falls
    back into the neutral zone -- at which point same-day reversal opens a new
    position immediately if the fresh signal is already non-neutral in the new
    direction. A position still open at the end of `sized_target` is force-
    exited past the last index (the window-boundary flatten kb/fold_spec.md
    section 6 already requires) -- unlike a signal-driven exit, the position
    was standing through the final day too, so `exit_pos` here is `n`, not
    `n - 1`: `strategy_returns`' `[entry_pos:exit_pos)` slice must still
    include that last day's gross P&L, even though its exit cost lands on the
    last real index.
    """
    trades: list[Trade] = []
    direction = 0
    size = 0.0
    entry_pos: int | None = None

    values = sized_target.to_numpy()
    n = len(values)
    for i, t in enumerate(values):
        want_dir = 0 if t == 0 else (1 if t > 0 else -1)
        if direction != 0 and want_dir != direction:
            trades.append(Trade(entry_pos, i, direction, size))
            direction, size, entry_pos = 0, 0.0, None
            if want_dir != 0:  # same-day reversal, #14 round 3 Q12
                direction, size, entry_pos = want_dir, abs(t), i
        elif direction == 0 and want_dir != 0:
            direction, size, entry_pos = want_dir, abs(t), i
        # else: hold at the fixed entry size, or stay flat -- nothing changes

    if direction != 0:
        trades.append(Trade(entry_pos, n, direction, size))
    return trades


def strategy_returns(
    trades: list[Trade],
    sized_target: pd.Series,
    daily_return: pd.Series,
    entry_bps: float,
    exit_bps: float,
) -> pd.Series:
    """Per-origin strategy return series, net of costs.

    `sized_target` is only used to assert it shares `daily_return`'s index --
    `trades` (built from `sized_target` by `simulate_single_position`) carries
    no reference back to it, so a caller passing a differently-aligned return
    series would otherwise index into the wrong sessions silently.

    Gross P&L accrues over `daily_return[entry_pos:exit_pos]` (the sessions
    actually held). Cost is charged as a return drag, linear in size, on the
    entry day and again on the exit day (clipped to the last real index, since
    a window-boundary force-exit's `exit_pos` can be one past it) -- a
    same-day reversal is two trades, so it is charged both an exit leg and an
    entry leg that day, not one discounted round trip (kb/evaluation_spec.md
    section 3).
    """
    if not sized_target.index.equals(daily_return.index):
        raise ValueError("sized_target and daily_return must share the same origin index")
    n = len(daily_return)
    raw = daily_return.to_numpy()
    out = np.zeros(n, dtype=float)
    for tr in trades:
        signed = tr.direction * tr.size
        end = min(tr.exit_pos, n)
        if tr.entry_pos < end:
            out[tr.entry_pos : end] += signed * raw[tr.entry_pos : end]
        out[tr.entry_pos] -= (entry_bps / 1e4) * tr.size
        exit_at = min(tr.exit_pos, n - 1)
        out[exit_at] -= (exit_bps / 1e4) * tr.size
    return pd.Series(out, index=daily_return.index)


def baseline_returns(daily_return: pd.Series, entry_bps: float, exit_bps: float) -> pd.Series:
    """Literal buy-and-hold over the given window: full size, entered once,
    held to the window's end, cost on entry and on the same window-boundary
    flatten the models are also charged for (kb/evaluation_spec.md section 4;
    `kb/evaluation_spec.json`'s `baseline.cost_note`). `daily_return` must
    already be sliced to the evaluation window."""
    returns = daily_return.astype(float).copy()
    if len(returns):
        returns.iloc[0] -= entry_bps / 1e4
        returns.iloc[-1] -= exit_bps / 1e4
    return returns


# --- Metrics (kb/evaluation_spec.md section 5) -------------------------------


def sharpe(returns: pd.Series, sessions_per_year: int) -> float:
    """Annualized Sharpe, net of costs. Zero (not NaN/inf) on a zero-vol series."""
    sigma = returns.std(ddof=1)
    if not sigma or np.isnan(sigma):
        return 0.0
    return float(returns.mean() / sigma * np.sqrt(sessions_per_year))


def cumulative_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def directional_hit_rate(position_direction: pd.Series, actual_direction: pd.Series) -> float:
    """Fraction of non-neutral-signal origins where the standing signal's
    direction matched the realized one-session direction. A diagnostic, not
    scored on a fixed horizon, since single-position holds run variable length.

    Also excludes origins where the realized direction itself is flat (zero) --
    there is no direction to have been right or wrong about that day, so it
    would otherwise always count as a miss regardless of the position held.
    """
    mask = (position_direction != 0) & (actual_direction != 0)
    if not mask.any():
        return float("nan")
    return float((position_direction[mask] == actual_direction[mask]).mean())


def horizon_rmse(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-horizon-step RMSE over an (n, horizon) prediction/actual pair."""
    return np.sqrt(np.mean((pred - actual) ** 2, axis=0))


# --- Reporting (kb/evaluation_spec.md section 6) -----------------------------


@dataclass(frozen=True)
class Scorecard:
    sharpe: float
    cumulative_return: float
    max_drawdown: float
    hit_rate: float
    n_sessions: int


def scorecard(
    returns: pd.Series,
    position_direction: pd.Series,
    actual_direction: pd.Series,
    sessions_per_year: int,
) -> Scorecard:
    return Scorecard(
        sharpe=sharpe(returns, sessions_per_year),
        cumulative_return=cumulative_return(returns),
        max_drawdown=max_drawdown(returns),
        hit_rate=directional_hit_rate(position_direction, actual_direction),
        n_sessions=len(returns),
    )


def report(
    returns: pd.Series,
    position_direction: pd.Series,
    actual_direction: pd.Series,
    fold_id: pd.Series,
    sessions_per_year: int,
) -> dict[str, Scorecard | dict[object, Scorecard]]:
    """Per-fold plus aggregate (kb/fold_spec.md's folds/holdout), per section 6."""
    for name, series in (
        ("position_direction", position_direction),
        ("actual_direction", actual_direction),
        ("fold_id", fold_id),
    ):
        if not series.index.equals(returns.index):
            raise ValueError(f"{name} must share returns' index")

    per_fold = {}
    for fid in fold_id.unique():
        mask = (fold_id == fid).to_numpy()
        per_fold[fid] = scorecard(
            returns[mask], position_direction[mask], actual_direction[mask], sessions_per_year
        )
    aggregate = scorecard(returns, position_direction, actual_direction, sessions_per_year)
    return {"aggregate": aggregate, "per_fold": per_fold}


def main() -> None:
    spec = load_spec()
    print(
        f"neutral_zone={spec.neutral_zone}  vol_window={spec.vol_window}  "
        f"entry_bps={spec.entry_bps}  exit_bps={spec.exit_bps}  "
        f"baseline_entry_bps={spec.baseline_entry_bps}  "
        f"sessions_per_year={spec.sessions_per_year}"
    )

    # -- target_size: neutral zone and clip --------------------------------
    raw = pd.Series([0.05, 0.5, -2.0, -0.1])
    sized = target_size(raw, spec.neutral_zone)
    assert sized.tolist() == [0.0, 0.5, -1.0, -0.1], sized.tolist()
    print("target_size: neutral zone and clip verified")

    # -- realized_vol / raw_signal: the vol-normalized sizing pipeline ------
    # -- (section 1) -- verified against an independently hand-derived std, --
    # -- not by trusting pandas' own rolling implementation ------------------
    returns6 = pd.Series([0.0, 0.02, -0.02, 0.02, -0.02, 0.02])
    vol = realized_vol(returns6, window=2, horizon=spec.horizon)
    assert np.isnan(vol.iloc[0])
    # std of 2 points a, b with ddof=1 is |a - b| / sqrt(2)
    expected_vol_1 = abs(0.0 - 0.02) / np.sqrt(2) * np.sqrt(spec.horizon)
    expected_vol_2 = abs(0.02 - (-0.02)) / np.sqrt(2) * np.sqrt(spec.horizon)
    assert np.isclose(vol.iloc[1], expected_vol_1)
    assert np.isclose(vol.iloc[2], expected_vol_2)

    cum_pred = pd.Series([0.05] * 6)
    raw6 = raw_signal(cum_pred, vol)
    assert np.isclose(raw6.iloc[1], 0.05 / expected_vol_1)
    assert np.isclose(raw6.iloc[2], 0.05 / expected_vol_2)

    sized6 = target_size(raw6, spec.neutral_zone)
    assert np.isclose(sized6.iloc[1], 1.0), "raw > 1 must clip"
    assert np.isclose(sized6.iloc[2], raw6.iloc[2]), "raw within [-1, 1] and outside the neutral zone passes through"
    print("realized_vol / raw_signal: vol-normalized sizing pipeline verified end-to-end")

    # -- simulate_single_position: entries, holds, flips, reversal, --------
    # -- neutral-drop exit, and the window-boundary force-flatten ----------
    sized_target = pd.Series([0.0, 0.0, 0.3, 0.6, -0.4, 0.2, 0.0, -0.5])
    trades = simulate_single_position(sized_target)
    expected = [
        Trade(2, 4, 1, 0.3),  # entered on the 0.3 signal, size held through the 0.6 day
        Trade(4, 5, -1, 0.4),  # same-day reversal out of the long above
        Trade(5, 6, 1, 0.2),  # same-day reversal again
        Trade(7, 8, -1, 0.5),  # entered on the last bar, force-flattened past it -- still held through it
    ]
    assert trades == expected, trades
    print(f"simulate_single_position: {len(trades)} trades match the hand-derived sequence")

    # -- strategy_returns: gross P&L plus entry/exit cost drag, including --
    # -- the same-day double charge on a reversal, and the force-flattened --
    # -- last trade still earning its final day's return -------------------
    daily_return = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.00, 0.015])
    returns = strategy_returns(trades, sized_target, daily_return, spec.entry_bps, spec.exit_bps)
    expected_returns = np.array([0.0, 0.0, 0.0087, 0.003, 0.0033, 0.0034, -0.0002, -0.0085])
    assert np.allclose(returns.to_numpy(), expected_returns, atol=1e-9), returns.tolist()
    print("strategy_returns: gross P&L and cost drag match a hand-derived series")

    try:
        strategy_returns(trades, sized_target, daily_return.iloc[:-1], spec.entry_bps, spec.exit_bps)
        raise AssertionError("mismatched index must raise")
    except ValueError:
        pass
    print("strategy_returns: rejects a daily_return misaligned with sized_target")

    # -- baseline_returns: entered once, cost on entry and on the window- --
    # -- boundary flatten (kb/evaluation_spec.json's baseline.cost_note) ---
    window = daily_return.iloc[2:7]
    base = baseline_returns(window, spec.baseline_entry_bps, spec.exit_bps)
    expected_base = np.array([0.03 - 0.0010, 0.01, -0.01, 0.02, 0.00 - 0.0010])
    assert np.allclose(base.to_numpy(), expected_base, atol=1e-9), base.tolist()
    print("baseline_returns: single continuous long, entry and boundary-flatten cost verified")

    # -- metrics --------------------------------------------------------
    flat = pd.Series([0.001, 0.001, 0.001])
    assert sharpe(flat, spec.sessions_per_year) == 0.0, "zero-vol series must not divide by zero"

    hand = pd.Series([0.02, -0.01, 0.015])
    expected_sharpe = hand.mean() / hand.std(ddof=1) * np.sqrt(spec.sessions_per_year)
    assert np.isclose(sharpe(hand, spec.sessions_per_year), expected_sharpe)

    dd_series = pd.Series([0.1, -0.2, 0.05])
    assert np.isclose(max_drawdown(dd_series), -0.2), max_drawdown(dd_series)
    assert np.isclose(cumulative_return(dd_series), 1.1 * 0.8 * 1.05 - 1.0)
    print("sharpe / max_drawdown / cumulative_return verified against hand calculations")

    pos_dir = pd.Series([1, 0, -1, 1])
    act_dir = pd.Series([1, 1, -1, -1])
    assert np.isclose(directional_hit_rate(pos_dir, act_dir), 2 / 3)
    assert np.isnan(directional_hit_rate(pd.Series([0, 0]), pd.Series([1, -1])))
    # a flat realized day (actual_direction == 0) has no direction to have
    # been right or wrong about, so it must not count as a miss
    assert directional_hit_rate(pd.Series([-1, 1]), pd.Series([0, -1])) == 0.0
    print("directional_hit_rate: masks neutral signals and flat realized days")

    pred = np.array([[0.01, 0.02], [0.03, 0.01]])
    actual = np.zeros_like(pred)
    rmse = horizon_rmse(pred, actual)
    assert np.allclose(rmse, [np.sqrt(0.0005), np.sqrt(0.00025)])
    print("horizon_rmse verified against hand calculation")

    # -- report: per-fold and aggregate --------------------------------
    fold_id = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    rpt = report(returns, sized_target.apply(np.sign), np.sign(daily_return), fold_id, spec.sessions_per_year)
    assert rpt["aggregate"].n_sessions == 8
    assert rpt["per_fold"]["A"].n_sessions == 4
    assert rpt["per_fold"]["B"].n_sessions == 4
    assert np.isclose(
        rpt["aggregate"].cumulative_return,
        cumulative_return(returns),
    )
    print("report: per-fold and aggregate scorecards verified consistent with the full series")

    print("\nAll self-checks passed. See kb/evaluation_spec.md for the full protocol.")


if __name__ == "__main__":
    main()
