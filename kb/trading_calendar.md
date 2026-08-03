# Trading-day calendar

Resolves [#4](https://github.com/jenujari/prdict-pov-v1/issues/4). Generated 2026-08-03 by `scripts/build_trading_calendar.py`; read via `prdict/trading_calendar.py`.

Map decision 1 puts every window on a **trading-day index**. History can be read off the data. The forward block cannot, because that is exactly where the prices are missing — so it needs a rule.

## The index

| Segment | Rule | Sessions | Range |
|---------|------|----------|-------|
| History | `c` non-null | 6517 | 2000-01-03 → 2026-06-30 |
| Forward | weekday minus frozen holidays | 146 | 2026-07-01 → 2027-01-29 |
| **Total** | | **6663** | 2000-01-03 → 2027-01-29 |

History is **observed, never derived**. The rule below applies only past 2026-06-30. Nothing re-derives a date the data already answers.

## Forward holidays

exchange_calendars 4.13.2 XBOM through 2026-12-31; 2027-01-26 hand-added (Republic Day, past XBOM's bound).

| Date | Day |
|------|-----|
| `2026-09-14` | Mon |
| `2026-10-02` | Fri |
| `2026-10-20` | Tue |
| `2026-11-10` | Tue |
| `2026-11-24` | Tue |
| `2026-12-25` | Fri |
| `2027-01-26` | Tue |

The list is **frozen and checked in**, not computed at run time: the index is byte-reproducible against a pinned model, and a date that turns out wrong is fixed by editing one line here and regenerating. `exchange_calendars` is a one-off generator, not a pipeline dependency — it is absent from `container/requirements.in` on purpose.

## Why not the bare weekday rule

Replayed over the observed history, weekday-only disagrees with the truth at:

| Window | Weekdays | Closures missed | Error rate |
|--------|----------|-----------------|------------|
| 2000–2026H1 | 6912 | 397 | 5.74% |
| excluding 2008 | 6650 | 360 | 5.41% |

That is 10–19 missed closures a year, so roughly 8 phantom rows across the forward block. The chosen rule was validated the same way: XBOM against 2006H2–2026H1 disagrees on **18 of 6906 days (0.26%)** once 2008 is set aside — a twentyfold improvement, and the residue is almost entirely Muhurat and special Saturday sessions rather than ordinary holidays.

Hardcoding only the fixed national dates (`01-26`, `04-14`, `05-01`, `08-15`, `10-02`, `12-25`) was measured too: those account for 114 of 397 historical closures (28.7%), leaving ~3.9% error. Every lunar festival — Diwali, Holi, Eid, Janmashtami — is still missed, so it buys little.

Predicting closures from the astro features themselves was tested and **fails**: a gradient-boosted classifier over all 215 feature columns caught 6 of 71 held-out holidays, worse than the 25 of 71 that bare calendar month/day features get on their own. NSE holidays track the civil festival calendar, not planetary position, so the index cannot be recovered from the data set.

## Forward runway

An **origin** is the last trading day of the past-60 encoder block. It is usable when 30 sessions exist strictly after it, since the known-covariate block spans them.

| | Date |
|-|------|
| First forward origin (last observed close) | `2026-06-30` |
| **Last origin with a full future-30 block** | **`2026-12-16`** |
| Last origin needing only the 10-step target horizon | `2027-01-14` |
| Forward origins available | 117 |

The ephemeris ends 2027-01-31, and that is the binding constraint — not data availability, since every feature is known-future (#3). Forward inference therefore covers **117 origins**, `2026-06-30` → `2026-12-16`. Origins between there and `2027-01-14` are reachable only if the decoder is shortened to 10 steps — an open question in [#5](https://github.com/jenujari/prdict-pov-v1/issues/5).

## What a mis-called date costs

Nothing in training: the forward block has no target, so a wrong forward date cannot corrupt a fit. The damage is confined to inference, and it is **not** a mislabelled row — it is a shift.

- **Phantom session** (predicted open, actually closed). The row enters the covariate block, so every later step in that window is fed the astro state of the wrong trading day and the returns come out shifted one step late. A window containing one phantom is wrong from the phantom onward, not just at it.
- **Missing session** (predicted closed, actually open). The window skips a real day; predictions after it run one step early. Same shift, opposite sign. The unplaced Muhurat session is this case.

Because the effect is a shift rather than a point error, there is no partial credit and no way to patch a single date after the fact — the fix is to correct `FORWARD_HOLIDAYS`, regenerate, and re-run inference from the affected origin. At a measured 0.26% that is expected to be zero or one date across the forward block, and NSE publishes its list far enough ahead that a correction lands before the date does.

## Gaps in the observed history

Resolves [#25](https://github.com/jenujari/prdict-pov-v1/issues/25). The observed index is authoritative for *which dates the market was open* — except where the source is simply missing rows. A missing session does not leave a hole: under map decision 1 the index collapses to trading days, so the two sessions either side become neighbours and the step between them silently spans more than one session's move.

**Policy: the missing sessions are treated as ordinary trading holidays.** Nothing is filtered; all 6448 origins are trainable.

Three facts make this the right default rather than a concession:

1. **The step they produce already exists, in bulk.** 2008's missing sessions are mostly Fridays, so the surviving step is Thursday → Monday — the exact calendar spacing a genuine Friday holiday produces, of which the index holds hundreds. The target was never a uniform time interval: a normal weekend already makes one step span three calendar days against another's one.
2. **The distortion is below the noise floor.** A straddling-return test — is the step across a gap inflated? — scores the confirmed 2008 gaps at mean z 0.67 against 0.24 for randomly chosen real holidays, with only 9 of 29 above z=1. The effect is not separable from ordinary holiday steps even knowing where to look.
3. **Filtering costs the regime the hypothesis most needs.** 2008 is the sample's most extreme year (volatility rank 1 of 27, 45.8% annualised, 38 sessions beyond ±4%). Excluding windows there trains a model that has never encoded those planetary configurations, and so cannot recognise them recurring.

What is given up: **147 origins (2.3%)** have one step of their 10-step target carrying roughly one extra session's move. Since the evaluation is a trading simulation driven by return magnitude, that exposure is recorded here rather than waved off — but it is one step in ten, in 2.3% of origins.

The alternatives are kept in `kb/trading_calendar.json` so an ablation can reproduce them without re-deriving anything: filtering on the target span leaves 6301 origins, and additionally requiring a clean encoder leaves 6167 — the latter retaining only 9 of 2008's 38 >4% sessions in any encoder block, against all 38 under the chosen policy. `cal.spans_gap(first, last)` applies either.

2008 is missing 21 sessions against XBOM's 246 — 29 dates XBOM calls sessions, 22 of them Fridays, spanning 2008-03-28 to 2008-11-28. Not a crash closure (the deficit starts in March and peaks in July; the crash days are all present) and not a date shift (shifting by +-1 does not improve agreement). Unrepairable: the rows carry all 235 astro columns but no price, and the map rules out new data sources.

Two hypotheses were tested and rejected. **Crash closures**: the deficit starts in March and peaks in July, months before Lehman, and the crash sessions themselves (`2008-10-28` at −8.78%, `2008-10-31` at +6.99%) are all present — NSE used intraday circuit breakers, never whole-day closures. **A date shift**: shifting 2008 by ±1 business day scores 0.863 against 0.859 unshifted, no improvement, where 2007 and 2009 score 0.992 and 1.000 unshifted.

### How a gap is detected

By **systematic source failure**, not date by date:

1. A year with at least **3** XBOM mismatches. 2008 has 29; every other year has 0–2, so any threshold in 3–28 isolates it.
2. A run of at least **4** consecutive closed weekdays, which no Indian holiday pattern produces. This is the only detector reaching before XBOM's 2006-08-03 start, and it fires exactly once, on `2002-08-27`–`2002-08-30`.

That gives **33 gap dates** and **28 discontinuities** in the session index. They are identified so results can be interpreted and so an ablation can target them — not because anything is dropped.

Note the gap dates are absent from the trading-day index entirely — no close means no row — so **their own** astro states are never encoded under any policy, including this one. Only their neighbours are. Encoding them would require a row with no target, which contradicts map decision 1.

### Known residue, deliberately unflagged

11 isolated XBOM mismatches sit outside the flagged years:

```
2007-09-05  2007-12-18  2011-08-18  2013-11-15  2014-02-19  2019-02-18  2022-09-12  2023-06-28  2024-01-20  2025-02-01  2025-09-08
```

Each is individually indistinguishable from XBOM being wrong. A straddling-return test — does the step across the suspect date look inflated? — **cannot separate them from ordinary holidays, and cannot separate the confirmed 2008 gaps either** (mean z 0.67 for 2008 against 0.24 for random holidays). With no evidence to stand on, flagging all 11 would cost **16.2%** of trainable origins against 3.3% for the gaps above. Two of them are Saturdays, where the source consistently omits NSE's special sessions. Set `GAP_YEAR_THRESHOLD = 1` to include them.

`2014-10-02`–`2014-10-06` was checked and is **genuine** — data and XBOM agree exactly (Gandhi Jayanti, Dussehra, Bakrid).

### Consequence for the folds

Since nothing is filtered, [#10](https://github.com/jenujari/prdict-pov-v1/issues/10) inherits **no** extra fold constraint — the session index is contiguous by construction and a gap is just another holiday. The one thing #10 should know is that 2008 carries the sample's densest run of them, so a fold boundary landing inside `2008-03-28`–`2008-11-28` sits in the least reliable stretch of the history. `cal.discontinuities` and `cal.spans_gap(first, last)` are available if the fold design wants to avoid it.
