# Target construction

Resolves [#8](https://github.com/jenujari/prdict-pov-v1/issues/8). Defined and read by `prdict/targets.py`; nothing is generated ahead of time, because the target is a pure function of the close series and the trading-day index of [#4](https://github.com/jenujari/prdict-pov-v1/issues/4).

## Definition

For an origin at session index `i`, the label is the vector

```
y[k] = log(C_{i+k} / C_{i+k-1})        k = 1 … 10
```

where every index steps along the **trading-day index**, never the calendar. So `y[1] = log(C_{i+1} / C_i)`: the origin `t` is the last session whose close the model may see, and the first predicted move is the one immediately after it. `y[10]` is the move into the tenth session ahead.

One array, shape **(6448, 10)**. Map decision 2 as amended by [#6](https://github.com/jenujari/prdict-pov-v1/issues/6) — both XGBoost and the TFT take the same step vector, so there is no second cumulative variant to keep in sync. Cumulative returns are `cumsum` along axis 1 and are derived at read time (`Targets.cumulative()`), never stored.

| | |
|-|-|
| Shape | `(6448, 10)` |
| Mean / std | `+0.000427` / `0.013561` |
| Min / max | `−0.1390` / `+0.1633` |
| NaN | none |
| Clipping | none applied |

## Usable origins

| | Date | Why |
|-|------|-----|
| **First usable origin** | **`2000-03-29`** | 60th session. The series starts `2000-01-03`; the encoder needs 60 sessions up to and including the origin. |
| **Last trainable origin** | **`2026-06-15`** | 10 sessions before the last observed close (`2026-06-30`), so the whole 10-step label is observed. |
| Trainable origins | **6448** | Contiguous — nothing is filtered. |
| First forward origin | `2026-06-30` | The last observed close; its forward returns are already unknown. |
| Last forward origin | `2026-12-16` | Bound by the ephemeris (#4), not by prices. |

Both dates are computed, not asserted — `cal.trainable_origins()` in `prdict/trading_calendar.py` derives them from the calendar, and `prdict/targets.py` re-verifies the alignment against the raw closes at three origins on every run.

Nothing is filtered on the 2008 gaps: [#25](https://github.com/jenujari/prdict-pov-v1/issues/25) settles that the missing sessions are ordinary trading holidays, so all 6448 origins are trainable and **147 (2.3%)** carry one step spanning roughly one extra session's move.

## TFT training target — 30-step extension

Amended by [#36](https://github.com/jenujari/prdict-pov-v1/issues/36), from [`docs/adr/0002`](../docs/adr/0002-tft-decodes-30-scores-10.md). The **scored** target above is unchanged: `(n, 10)`, and it is the only quantity any model is compared on.

The TFT's decoder spans the whole 30-session future block (`max_prediction_length = future = 30`), so for *training* it needs a 30-step label. `build(cal, steps=cal.future)` produces it — the same definition, run out to 30 steps:

| | 10-step scored | 30-step TFT training |
|-|-|-|
| Shape | `(6448, 10)` | `(6428, 30)` |
| Origins | `2000-03-29` → `2026-06-15` | `2000-03-29` → `2026-05-15` |
| Reason for the tail | needs `C_{i+10}` | needs `C_{i+30}` |

The 30-step build drops **20** origins at the tail — the ones without a full 30-step label. They sit in the final holdout region, which is *scored* (needs only the 10-step actual), not *trained*, so no fold's training range actually loses an origin; the real fold training blocks end years earlier. The self-check asserts the first 10 steps of the 30-step target coincide cell-for-cell with the scored target, so the two views cannot drift. `elapsed` extends the same way (`elapsed_1..30`).

## Source ordering

`nft50.csv` is stored **newest-first**. `restrict()` now sorts ascending, so no downstream stage has to know — but the failure mode is worth recording, because it is silent: reading the close series in file order negates every return and still produces a plausible-looking label matrix with the right shape, the right scale, and no NaNs. Nothing but a sign check catches it.

## Elapsed calendar time between steps

A step is one *trading* day, but that is not one unit of time — a step across a weekend spans three calendar days, and one across a holiday stretch up to seven. Measured over the 6516 observed steps:

| Calendar days spanned | n | mean \|r\| | std ratio vs 1d | √t |
|---|---|---|---|---|
| 1 | 4945 | 0.0088 | 1.000 | 1.000 |
| 2 | 182 | 0.0111 | 1.209 | 1.414 |
| 3 | 1210 | 0.0101 | 1.262 | 1.732 |
| 4 | 162 | 0.0149 | 1.664 | 2.000 |
| 5 | 15 | 0.0216 | 2.555 | 2.236 |
| 6–7 | 2 | 0.0132 | — | — |

The effect is **real and monotone** — a 5-day step carries 2.6× the volatility of a 1-day step — but it is **not** √t. Weekend steps come in well under √t (1.21 and 1.26 against 1.41 and 1.73), the long-standing weekend-variance result; only the 4- and 5-day steps reach or exceed it.

**Decision: the target is left raw, and elapsed calendar days becomes a per-step known-future covariate.**

- Normalising by √t would apply the wrong exponent — over-correcting weekends by ~16% — and it makes every prediction require an un-normalising step before the trading simulation of map decision 8. The scorecard trades actual moves, so the label should be the actual move.
- Leaving it alone with no covariate discards a measured 2.7× effect the model could condition on.
- The covariate is **free**: the trading-day index is fixed and known-future through `2027-01-29` (#4), so `elapsed_1 … elapsed_10` is available at inference exactly as in training. `prdict/targets.py` emits it beside the label (`Targets.elapsed`) and `forward_elapsed()` supplies it for the 117 label-less forward origins.

24.1% of label cells span more than one calendar day.

**Consequence for the feature tickets.** `elapsed_days` is a decoder-side known-future covariate: a real column for the TFT's `time_varying_known_reals`, and 10 columns for XGBoost's flat design matrix. It is calendar-derived, not price-derived, so map decision 5 does not touch it.

### This also re-confirms #25

The 28 steps spanning a known 2008/2002 gap look inflated on their face — mean \|z\| 1.35 against 0.79 for the rest. Conditioned on elapsed calendar days, that difference almost vanishes:

| Elapsed | Gap steps | mean \|z\| | Ordinary peers |
|---|---|---|---|
| 2d | 6 | 0.964 | 0.970 |
| 4d | 17 | 1.440 | 1.099 |
| 5d | 4 | 1.513 | 1.054 |

A gap step is mostly just a long step, which is exactly #25's claim — the surviving Thursday → Monday move has the calendar spacing of a genuine Friday holiday, of which the index holds hundreds. The residual at 4–5 days is 21 steps concentrated in 2008, the sample's most volatile year.

## Outliers

**Decision: no winsorisation, no clipping — the target is used raw.**

The tails are fat. Kurtosis is **10.45**, skew −0.45, and `|r| > 5%` on **1.03%** of steps, which touches **5.85%** of the 10-step rows.

| Threshold | Steps | Share | Years |
|---|---|---|---|
| \|r\| > 5% | 67 | 1.03% | 2000, 01, 04, 06, 07, 08, 09, 15, 19, 20, 24 |
| \|r\| > 6% | 30 | 0.46% | 2000, 01, 04, 06, 08, 09, 15, 20, 24 |
| \|r\| > 8% | 8 | 0.12% | 2004, 2008, 2009, 2020 |
| \|r\| > 10% | 3 | 0.05% | 2004, 2009, 2020 |

The three largest are `2009-05-18` (+16.3%, election result), `2020-03-23` (−13.9%, COVID lockdown), `2004-05-17` (−13.1%, election result).

Why keep them: **the evaluation is a trading simulation** scored on cumulative return, Sharpe and max drawdown (map decision 8). Those days *are* the P&L and the drawdown. Clipping the label trains the model to under-predict precisely the moves the scorecard is most sensitive to, and it does so asymmetrically — clipping at ±5% removes 0.97 of 60.5 total absolute return, almost all of it from a handful of days that decide whether a strategy survives 2008 and 2020.

Tail sensitivity in *optimisation* is a separate problem with a separate fix. If a fit is destabilised by these days, the lever is the **loss** — Huber, or the quantile loss the TFT uses by default — which is a model-ticket knob and leaves the label, and therefore the scorecard, untouched. Nothing distorts the data on the way in.

## Overlapping windows — what #10 inherits

Consecutive origins share **9 of 10** label days. Samples are therefore **not** independent, and the CV design of [#10](https://github.com/jenujari/prdict-pov-v1/issues/10) has to account for it:

| Separation between two origins | What they share |
|---|---|
| `< 10` sessions | At least one **scored** label day → direct label leakage |
| `< 30` sessions | At least one **TFT training** label day (30-step target) |
| `< 60` sessions | At least one encoder row |
| `< 69` sessions | Any row at all |

- **Purge ≥ 10 sessions** is the floor for the 10-step scored target; below it a training origin's label literally contains a test origin's label days.
- **Purge = 30 sessions** is the floor once the TFT trains on the 30-step label (`docs/adr/0002`) — a 30-step training label reaches 30 sessions forward, so anything closer shares a scored return. Since the comparison runs on one fold geometry, **#10 adopts purge 30 for both models** (amended by [#36](https://github.com/jenujari/prdict-pov-v1/issues/36)); the extra cost to the 10-step XGBoost is ~20 origins per boundary (~0.3%).
- **89 sessions** (60 encoder + 29 label) would additionally buy complete *row-level* separation, but PR #34 settled that the encoder overlap is harmless — every feature is known-future (#3) — so the fold spec sits on the label floor, not the row floor.

Note the encoder overlap is a weaker form of leakage than the label overlap — shared *inputs* between folds are ordinary in time-series CV; shared *labels* are not.

## Reading the target

```python
from prdict.targets import build

t = build()                 # loads the calendar and closes
t.origins                   # DatetimeIndex, 6448
t.y                         # (6448, 10) step log returns
t.elapsed                   # (6448, 10) calendar days per step, int16
t.cumulative()              # (6448, 10) derived, not stored
t.frame                     # wide DataFrame: y_1..y_10, elapsed_1..elapsed_10
```

```sh
uv run python -m prdict.targets    # self-check, incl. the r_1 = C_t+1/C_t assertion
```
