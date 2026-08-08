"""Build the 10-step log-return target and its per-step elapsed-time covariate.

The target is a pure function of the close series and the trading-day index, so
unlike the column spec and the calendar there is nothing to generate ahead of
time — this module *is* the definition. See `kb/target_spec.md` for why each
choice is what it is.

    uv run python -m prdict.targets    # self-check against nft50.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prdict.trading_calendar import CSV_PATH, TradingCalendar, load_calendar, restrict


@dataclass(frozen=True)
class Targets:
    """The label matrix and the covariate that says how much time each step spans.

    `horizon` is the number of steps this array holds. The scored target — the
    one both models are compared on — is the 10-step build. The TFT also trains
    on a 30-step build (`steps=cal.future`), because its decoder spans the whole
    future block; only its first 10 steps are ever scored (`docs/adr/0002`).
    """

    origins: pd.DatetimeIndex
    y: np.ndarray
    elapsed: np.ndarray
    horizon: int

    @property
    def frame(self) -> pd.DataFrame:
        """Wide view, one row per origin — `y_1..y_h` beside `elapsed_1..elapsed_h`."""
        step = range(1, self.horizon + 1)
        return pd.DataFrame(
            np.hstack([self.y, self.elapsed]),
            index=self.origins,
            columns=[f"y_{k}" for k in step] + [f"elapsed_{k}" for k in step],
        )

    def cumulative(self) -> np.ndarray:
        """Cumulative log return at each horizon — derived, never stored.

        Map decision 2 keeps exactly one target array; #6 settles that both
        models take the step vector. Anything wanting the cumulative form for
        reporting builds it here rather than carrying a parallel label.
        """
        return np.cumsum(self.y, axis=1)


def close_series(cal: TradingCalendar, csv_path=CSV_PATH) -> pd.Series:
    """Closes on the trading-day index, ascending, NaN over the forward block.

    The source CSV is stored newest-first, so this sorts rather than trusting
    file order — an inverted series would silently negate every return.
    """
    frame = pd.read_csv(csv_path, usecols=["record_date", "c"], parse_dates=["record_date"])
    frame = restrict(frame, cal)
    return frame.set_index("record_date")["c"]


def step_returns(cal: TradingCalendar, closes: pd.Series | None = None) -> pd.Series:
    """One-session log returns, indexed by the session the move *lands on*.

    `returns[d] = log(C_d / C_prev)`, so the value at session `d` is the move
    from the previous session's close into `d`. The first session has no
    predecessor and is dropped.
    """
    closes = close_series(cal) if closes is None else closes
    observed = closes.iloc[: cal.n_history]
    return np.log(observed).diff().iloc[1:]


def elapsed_days(cal: TradingCalendar) -> pd.Series:
    """Calendar days each session step spans, indexed the same way as the returns.

    Known for the forward block too — the trading-day index is fixed by #4 —
    so the same covariate is available at inference as in training.
    """
    sessions = cal.sessions
    days = (sessions[1:] - sessions[:-1]).days
    return pd.Series(np.asarray(days, dtype=np.int16), index=sessions[1:])


def build(
    cal: TradingCalendar | None = None,
    closes: pd.Series | None = None,
    steps: int | None = None,
) -> Targets:
    """The (n, steps) label matrix over every trainable origin, plus its elapsed covariate.

    An origin is the last session whose close the model may see; step `k` of its
    label is the move into the k-th session after it, so `y[:, 0]` is
    `log(C_{t+1} / C_t)`.

    `steps` defaults to `cal.horizon` (the scored 10-step target). Pass
    `cal.future` for the TFT's 30-step training target — a longer label over a
    smaller origin set (the tail loses origins lacking a full 30-step label).
    """
    cal = load_calendar() if cal is None else cal
    steps = cal.horizon if steps is None else steps
    returns = step_returns(cal, closes)
    elapsed = elapsed_days(cal)

    origins = cal.trainable_origins(reach=steps)
    # Every origin's label spans the `steps` sessions strictly after it, so the
    # rows are contiguous slices of the return series starting one step along.
    first = returns.index.get_indexer([origins[0]])[0] + 1
    if first < 0:
        raise ValueError(f"origin {origins[0].date()} is not on the return index")

    y = np.lib.stride_tricks.sliding_window_view(returns.to_numpy(float), steps)
    e = np.lib.stride_tricks.sliding_window_view(elapsed.to_numpy(), steps)
    y = y[first : first + len(origins)]
    e = e[first : first + len(origins)]

    if np.isnan(y).any():
        raise ValueError("target matrix has NaNs — trainable_origins() overran the closes")
    if len(y) != len(origins):
        raise ValueError(f"{len(origins)} origins but {len(y)} label rows")

    return Targets(origins=origins, y=y, elapsed=e, horizon=steps)


def forward_elapsed(cal: TradingCalendar) -> pd.DataFrame:
    """The elapsed covariate over the forward origins, which carry no label.

    Inference needs the same per-step covariate the fit was given; the calendar
    supplies it without any close.
    """
    elapsed = elapsed_days(cal)
    origins = cal.forward_origins()
    windows = np.lib.stride_tricks.sliding_window_view(elapsed.to_numpy(), cal.horizon)
    first = elapsed.index.get_indexer([origins[0]])[0] + 1
    return pd.DataFrame(
        windows[first : first + len(origins)],
        index=origins,
        columns=[f"elapsed_{k}" for k in range(1, cal.horizon + 1)],
    )


def main() -> None:
    cal = load_calendar()
    t = build(cal)

    print(f"origins  : {len(t.origins)}  {t.origins[0].date()} → {t.origins[-1].date()}")
    print(f"y        : {t.y.shape}  no NaN, no clipping applied")
    print(f"mean {t.y.mean():+.6f}  std {t.y.std():.6f}  "
          f"min {t.y.min():+.4f}  max {t.y.max():+.4f}")

    # Alignment is the whole ticket, so assert it against the raw closes rather
    # than trusting the slicing above.
    closes = close_series(cal)
    for probe in (0, len(t.origins) // 2, len(t.origins) - 1):
        origin = t.origins[probe]
        i = cal.position(origin)
        expected = np.log(closes.iloc[i + 1] / closes.iloc[i])
        assert np.isclose(t.y[probe, 0], expected), origin
        last = np.log(closes.iloc[i + cal.horizon] / closes.iloc[i + cal.horizon - 1])
        assert np.isclose(t.y[probe, -1], last), origin
    print(f"alignment: r_1 = C_t+1/C_t verified at 3 origins, r_{cal.horizon} at the far end")

    cum = t.cumulative()
    print(f"cumulative: derived {cum.shape}, h{cal.horizon} mean {cum[:, -1].mean():+.5f}")

    e = t.elapsed
    print(f"\nelapsed  : {e.shape}  {e.min()}–{e.max()} calendar days per step")
    print(f"  steps spanning >1 day: {100 * (e > 1).mean():.1f}% of cells")
    fwd = forward_elapsed(cal)
    print(f"  forward origins carrying the covariate: {len(fwd)} "
          f"{fwd.index[0].date()} → {fwd.index[-1].date()}")

    tail = np.abs(t.y) > 0.05
    print(f"\ntails    : |r|>5% on {100 * tail.mean():.3f}% of cells, "
          f"{100 * tail.any(1).mean():.2f}% of rows — kept (see kb/target_spec.md)")

    # TFT training target: the decoder spans the full future block, so it trains
    # on a 30-step label over a smaller origin set. Only its first 10 steps are
    # ever scored — verify they coincide with the scored target (docs/adr/0002).
    tft = build(cal, steps=cal.future)
    dropped = len(t.origins) - len(tft.origins)
    print(f"\ntft target: {tft.y.shape}  steps={cal.future}  "
          f"{tft.origins[0].date()} → {tft.origins[-1].date()}")
    print(f"  origins  : {len(tft.origins)} ({dropped} fewer than the 10-step set — "
          f"the tail lacking a full {cal.future}-step label)")
    assert np.allclose(tft.y[: len(tft.origins), : cal.horizon],
                       t.y[: len(tft.origins)]), "first-10 of 30-step target must match scored"
    assert np.allclose(tft.elapsed[: len(tft.origins), : cal.horizon],
                       t.elapsed[: len(tft.origins)]), "first-10 elapsed must match"
    print(f"  first {cal.horizon} steps coincide with the scored target: verified")


if __name__ == "__main__":
    main()
