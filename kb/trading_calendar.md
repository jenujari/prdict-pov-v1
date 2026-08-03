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

## Known defect: 2008

2008 holds ~22 spurious closures (22 of 37 weekday closures are Fridays; 225 sessions against a ~246 norm). Treated as a data defect, excluded from every error rate here, not repaired. It is **not** repaired here — 2008 sits inside the training history, so ~22 phantom closures collapse real consecutive sessions into neighbours and quietly corrupt every window that crosses them.
