"""Derive the trading-day index for nft50.csv, history and forward block alike.

Resolves wayfinder ticket #4. Map decision 1 puts the whole pipeline on a
trading-day index. For history that index is *observed* — a trading day is a row
where `c` is non-null. After 2026-06-30 every OHLC value is NaN, so the forward
index has to come from a rule instead.

The rule is: weekday, minus a frozen holiday list (`FORWARD_HOLIDAYS` below).
That list is checked in rather than computed at run time, so the index is
byte-reproducible forever and a wrong date is fixed by editing one line. It was
generated once from `exchange_calendars`' XBOM calendar (BSE and NSE observe the
same holidays) and hand-extended through January 2027, which is past XBOM's
2026-12-31 bound. Rerun with `--audit` to re-check it against XBOM if that
package happens to be installed; nothing in the pipeline imports it.

Writes kb/trading_calendar.json (machine-readable) and kb/trading_calendar.md.

    uv run python scripts/build_trading_calendar.py
    uv run python scripts/build_trading_calendar.py --audit
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "nft50.csv"
CAL_JSON = ROOT / "kb" / "trading_calendar.json"
CAL_MD = ROOT / "kb" / "trading_calendar.md"

KEY = "record_date"
TARGET_SOURCE = "c"

# The forward block: every date in the source data past the last observed close.
FORWARD_START = "2026-07-01"
FORWARD_END = "2027-01-31"

# Window arithmetic from map decision 1 — 60 past / 30 future / 10 ahead, all in
# trading days.
PAST = 60
FUTURE = 30
HORIZON = 10

# Weekdays in the forward block on which the market is closed.
#
# 2026-07-01 .. 2026-12-31 come from exchange_calendars 4.13.2, calendar XBOM.
# 2027-01-26 is Republic Day, a fixed national holiday that NSE has observed
# every year in the source data; it is past XBOM's last session (2026-12-31) and
# is the only January-2027 closure. NSE trades on 2027-01-01.
FORWARD_HOLIDAYS = [
    "2026-09-14",
    "2026-10-02",
    "2026-10-20",
    "2026-11-10",
    "2026-11-24",
    "2026-12-25",
    "2027-01-26",
]

# 2008 is a data defect, not a run of holidays: 29 of its 37 weekday closures
# are days XBOM calls sessions, 22 of them Fridays, and the year totals 225
# sessions against XBOM's 246. Every error rate below is reported both with and
# without it so the rule is not judged against corrupt truth. See #25.
BAD_YEARS = [2008]

# Dates where a session is missing from the source data, so the step across them
# spans more than one session's move. A window covering such a step is training
# on a discontinuity, so #25 drops those windows rather than those sessions.
#
# Detected as *systematic source failure*, not date by date. Two rules:
#
#   1. A year whose XBOM mismatches exceed GAP_YEAR_THRESHOLD — 2008 has 29,
#      every other year has 0-2, so any threshold in 3..28 isolates it.
#   2. A run of >= GAP_RUN_LENGTH consecutive closed weekdays, which no Indian
#      holiday pattern produces. This is the only detector that reaches before
#      XBOM's 2006-08-03 start, and it fires exactly once, on 2002-08-27..30.
#
# Eleven further isolated XBOM mismatches exist across 2007-2025 and are
# deliberately *not* flagged: individually they are indistinguishable from XBOM
# being wrong, a straddling-return test cannot separate them from ordinary
# holidays (it cannot separate the confirmed 2008 gaps either), and flagging
# them costs 16.2% of trainable origins against 3.3% for these. Two of them are
# Saturdays, where the source consistently omits NSE's special sessions. Listed
# in the spec as known residue; lower GAP_YEAR_THRESHOLD to 1 to include them.
GAP_YEAR_THRESHOLD = 3
GAP_RUN_LENGTH = 4

GAP_DATES = [
    "2002-08-27", "2002-08-28", "2002-08-29", "2002-08-30",
    "2008-03-28", "2008-04-17", "2008-05-16", "2008-05-30", "2008-06-13",
    "2008-06-20", "2008-06-27", "2008-07-04", "2008-07-07", "2008-07-09",
    "2008-07-11", "2008-07-18", "2008-07-25", "2008-08-08", "2008-08-14",
    "2008-08-22", "2008-09-02", "2008-09-12", "2008-09-19", "2008-09-26",
    "2008-10-01", "2008-10-24", "2008-10-27", "2008-10-29", "2008-11-12",
    "2008-11-14", "2008-11-21", "2008-11-26", "2008-11-28",
]

# Isolated XBOM mismatches left unflagged — recorded so the choice is visible.
UNFLAGGED_MISMATCHES = [
    "2007-09-05", "2007-12-18", "2011-08-18", "2013-11-15", "2014-02-19",
    "2019-02-18", "2022-09-12", "2023-06-28", "2024-01-20", "2025-02-01",
    "2025-09-08",
]


def observed_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """History's trading days, read off the data rather than derived by rule."""
    return pd.DatetimeIndex(frame.loc[frame[TARGET_SOURCE].notna(), KEY])


def forward_index() -> pd.DatetimeIndex:
    """The forward block's trading days: weekdays minus the frozen holidays."""
    weekdays = pd.date_range(FORWARD_START, FORWARD_END, freq="B")
    return weekdays.drop(pd.DatetimeIndex(FORWARD_HOLIDAYS))


def weekday_rule_error(frame: pd.DataFrame) -> dict:
    """How often a bare weekday rule would disagree with the observed history.

    This is the honest cost of the cheapest possible forward rule, measured on
    the only stretch where truth is visible.
    """
    hist = frame[frame[KEY] <= frame.loc[frame[TARGET_SOURCE].notna(), KEY].max()].copy()
    hist["is_weekday"] = hist[KEY].dt.dayofweek < 5
    hist["is_open"] = hist[TARGET_SOURCE].notna()
    hist["year"] = hist[KEY].dt.year

    def rate(sub: pd.DataFrame) -> dict:
        weekdays = sub[sub["is_weekday"]]
        closed = int((~weekdays["is_open"]).sum())
        weekend_open = int((~sub["is_weekday"] & sub["is_open"]).sum())
        return {
            "weekdays": len(weekdays),
            "weekday_closures_missed": closed,
            "weekend_sessions_missed": weekend_open,
            "error_rate": round(closed / len(weekdays), 5),
        }

    clean = hist[~hist["year"].isin(BAD_YEARS)]
    return {
        "all_history": rate(hist),
        "excluding_bad_years": rate(clean),
        "weekend_sessions": [
            d.date().isoformat()
            for d in hist.loc[~hist["is_weekday"] & hist["is_open"], KEY]
        ],
        "closures_per_year": (
            hist[hist["is_weekday"] & ~hist["is_open"]].groupby("year").size().to_dict()
        ),
    }


def gap_cost(history: pd.DatetimeIndex) -> dict:
    """What filtering on gaps *would* cost — recorded, not applied.

    #25 settles that the missing sessions are treated as ordinary trading
    holidays, so every origin is trainable. These numbers size the exposure and
    let an ablation reproduce the alternatives without re-deriving them.
    """
    gaps = pd.DatetimeIndex(GAP_DATES)
    positions = {int(p) for p in history.get_indexer(gaps, method="bfill") if p > 0}
    total = len(history) - PAST - HORIZON + 1
    candidates = range(PAST - 1, len(history) - HORIZON)

    def kept(lo_offset: int) -> int:
        return sum(
            1
            for i in candidates
            if not any(i + lo_offset < p <= i + HORIZON for p in positions)
        )

    target_only, whole_window = kept(0), kept(-PAST + 1)
    return {
        "n_gap_dates": len(gaps),
        "n_discontinuities": len(positions),
        "policy": "kept — missing sessions are treated as trading holidays",
        "trainable_origins": total,
        "exposure": {
            "origins_with_gap_in_target": total - target_only,
            "fraction": round((total - target_only) / total, 4),
        },
        "if_filtered_on_target": {
            "origins": target_only,
            "dropped": total - target_only,
        },
        "if_filtered_on_whole_window": {
            "origins": whole_window,
            "dropped": total - whole_window,
        },
    }


def build(frame: pd.DataFrame) -> dict:
    history = observed_index(frame)
    forward = forward_index()
    full = history.append(forward)

    # An origin is the last trading day of the past-60 encoder block. It needs
    # `FUTURE` sessions strictly after it for the known-covariate block, and
    # `PAST` sessions up to and including itself for the encoder.
    last_full = full[len(full) - FUTURE - 1]
    last_horizon_only = full[len(full) - HORIZON - 1]
    first_forward_origin = history[-1]  # last observed close: its target is unknown

    n_forward_origins = full.get_loc(last_full) - full.get_loc(first_forward_origin) + 1

    return {
        "generated": date.today().isoformat(),
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/4",
        "source": {
            "csv": CSV_PATH.name,
            "n_rows": len(frame),
            "first_date": frame[KEY].min().date().isoformat(),
            "last_date": frame[KEY].max().date().isoformat(),
        },
        "history": {
            "rule": "observed — a trading day is a row where `c` is non-null",
            "n_sessions": len(history),
            "first": history[0].date().isoformat(),
            "last": history[-1].date().isoformat(),
        },
        "forward": {
            "rule": "weekday minus the frozen holiday list",
            "provenance": "exchange_calendars 4.13.2 XBOM through 2026-12-31; "
            "2027-01-26 hand-added (Republic Day, past XBOM's bound)",
            "n_sessions": len(forward),
            "first": forward[0].date().isoformat(),
            "last": forward[-1].date().isoformat(),
            "holidays": FORWARD_HOLIDAYS,
            "sessions": [d.date().isoformat() for d in forward],
        },
        "index": {
            "n_sessions": len(full),
            "past": PAST,
            "future": FUTURE,
            "horizon": HORIZON,
        },
        "runway": {
            "last_origin_full_future_block": last_full.date().isoformat(),
            "last_origin_horizon_only": last_horizon_only.date().isoformat(),
            "first_forward_origin": first_forward_origin.date().isoformat(),
            "n_forward_origins": int(n_forward_origins),
        },
        "weekday_rule_error": weekday_rule_error(frame),
        "gaps": {
            "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/25",
            "policy": "the missing sessions are treated as ordinary trading "
            "holidays — nothing is filtered. The defect is recorded so results "
            "can be interpreted, and `spans_gap` remains available for an ablation.",
            "detection": {
                "year_threshold": GAP_YEAR_THRESHOLD,
                "run_length": GAP_RUN_LENGTH,
                "note": "systematic source failure only — a year with more XBOM "
                "mismatches than the threshold, or a run of consecutive closed "
                "weekdays no holiday pattern produces",
            },
            "dates": GAP_DATES,
            "unflagged_mismatches": UNFLAGGED_MISMATCHES,
            "unflagged_note": "isolated XBOM mismatches, individually "
            "indistinguishable from the reference being wrong; flagging them "
            "would cost 16.2% of trainable origins against 3.3% for the gaps",
            "cost": gap_cost(observed_index(frame)),
        },
        "known_gaps": {
            "bad_years": BAD_YEARS,
            "note": "2008 is missing 21 sessions against XBOM's 246 — 29 dates "
            "XBOM calls sessions, 22 of them Fridays, spanning 2008-03-28 to "
            "2008-11-28. Not a crash closure (the deficit starts in March and "
            "peaks in July; the crash days are all present) and not a date "
            "shift (shifting by +-1 does not improve agreement). Unrepairable: "
            "the rows carry all 235 astro columns but no price, and the map "
            "rules out new data sources.",
        },
        "muhurat": {
            "note": "NSE's ceremonial Diwali Muhurat session is a real trading "
            "day that the frozen list does not attempt to place. Two such "
            "weekend sessions exist in 27 years of history. If Muhurat 2026 "
            "falls on a weekend it is absent from the forward index; the cost "
            "is one missing session, not a shifted one."
        },
    }


def audit(spec: dict) -> None:
    """Re-derive the forward holidays from XBOM and diff against the frozen list."""
    try:
        import exchange_calendars as xc
    except ImportError:
        print("exchange_calendars not installed — skipping audit")
        return

    cal = xc.get_calendar("XBOM")
    covered_end = min(pd.Timestamp(FORWARD_END), cal.last_session)
    sessions = pd.DatetimeIndex(
        cal.sessions_in_range(FORWARD_START, covered_end)
    ).normalize()
    weekdays = pd.date_range(FORWARD_START, covered_end, freq="B")
    derived = sorted(d.date().isoformat() for d in set(weekdays) - set(sessions))
    frozen = [h for h in FORWARD_HOLIDAYS if h <= covered_end.date().isoformat()]

    print(f"\naudit against XBOM (covers to {cal.last_session.date()})")
    if derived == frozen:
        print(f"  frozen list matches XBOM on all {len(frozen)} dates")
    else:
        print(f"  MISMATCH\n    XBOM  : {derived}\n    frozen: {frozen}")
    print(f"  hand-filled past XBOM's bound: "
          f"{[h for h in FORWARD_HOLIDAYS if h not in frozen]}")

    # Re-derive the gap list from the two detection rules.
    frame = pd.read_csv(CSV_PATH, usecols=[KEY, TARGET_SOURCE], parse_dates=[KEY])
    last_close = frame.loc[frame[TARGET_SOURCE].notna(), KEY].max()
    hist = frame[frame[KEY] <= last_close].copy()
    hist["is_open"] = hist[TARGET_SOURCE].notna()

    xbom_start = cal.first_session
    covered = hist[hist[KEY] >= xbom_start]
    sessions = set(
        pd.DatetimeIndex(cal.sessions_in_range(xbom_start, last_close)).normalize()
    )
    mismatch = covered[~covered["is_open"] & covered[KEY].isin(sessions)]
    per_year = mismatch.groupby(mismatch[KEY].dt.year).size()
    bad_years = per_year[per_year >= GAP_YEAR_THRESHOLD]
    rule1 = mismatch[mismatch[KEY].dt.year.isin(bad_years.index)][KEY]

    weekdays = hist[hist[KEY].dt.dayofweek < 5].reset_index(drop=True)
    rule2, run = [], []
    for _, row in weekdays.iterrows():
        if row["is_open"]:
            if len(run) >= GAP_RUN_LENGTH:
                rule2 += run
            run = []
        else:
            run.append(row[KEY])
    if len(run) >= GAP_RUN_LENGTH:
        rule2 += run

    derived = sorted(d.date().isoformat() for d in set(rule1) | set(rule2))
    print(f"\n  gap rules: year threshold >={GAP_YEAR_THRESHOLD} "
          f"(flagged {list(bad_years.index)}), run length >={GAP_RUN_LENGTH}")
    if derived == sorted(GAP_DATES):
        print(f"  frozen GAP_DATES matches the rules on all {len(derived)} dates")
    else:
        only_derived = sorted(set(derived) - set(GAP_DATES))
        only_frozen = sorted(set(GAP_DATES) - set(derived))
        print(f"  MISMATCH\n    rules only : {only_derived}\n    frozen only: {only_frozen}")

    residue = sorted(
        d.date().isoformat() for d in set(mismatch[KEY]) - set(rule1)
    )
    print(f"  unflagged isolated mismatches ({len(residue)}): {residue}")


def render_markdown(spec: dict) -> str:
    err = spec["weekday_rule_error"]
    fwd, run = spec["forward"], spec["runway"]
    dow = {d: pd.Timestamp(d).strftime("%a") for d in fwd["holidays"]}

    lines = [
        "# Trading-day calendar",
        "",
        f"Resolves [#4]({spec['resolves']}). Generated {spec['generated']} by "
        "`scripts/build_trading_calendar.py`; read via `prdict/trading_calendar.py`.",
        "",
        "Map decision 1 puts every window on a **trading-day index**. History can be "
        "read off the data. The forward block cannot, because that is exactly where the "
        "prices are missing — so it needs a rule.",
        "",
        "## The index",
        "",
        "| Segment | Rule | Sessions | Range |",
        "|---------|------|----------|-------|",
        f"| History | `c` non-null | {spec['history']['n_sessions']} | "
        f"{spec['history']['first']} → {spec['history']['last']} |",
        f"| Forward | weekday minus frozen holidays | {fwd['n_sessions']} | "
        f"{fwd['first']} → {fwd['last']} |",
        f"| **Total** | | **{spec['index']['n_sessions']}** | "
        f"{spec['history']['first']} → {fwd['last']} |",
        "",
        "History is **observed, never derived**. The rule below applies only past "
        f"{spec['history']['last']}. Nothing re-derives a date the data already answers.",
        "",
        "## Forward holidays",
        "",
        f"{fwd['provenance']}.",
        "",
        "| Date | Day |",
        "|------|-----|",
        *[f"| `{h}` | {dow[h]} |" for h in fwd["holidays"]],
        "",
        "The list is **frozen and checked in**, not computed at run time: the index is "
        "byte-reproducible against a pinned model, and a date that turns out wrong is "
        "fixed by editing one line here and regenerating. `exchange_calendars` is a "
        "one-off generator, not a pipeline dependency — it is absent from "
        "`container/requirements.in` on purpose.",
        "",
        "## Why not the bare weekday rule",
        "",
        "Replayed over the observed history, weekday-only disagrees with the truth at:",
        "",
        "| Window | Weekdays | Closures missed | Error rate |",
        "|--------|----------|-----------------|------------|",
        f"| 2000–2026H1 | {err['all_history']['weekdays']} | "
        f"{err['all_history']['weekday_closures_missed']} | "
        f"{100 * err['all_history']['error_rate']:.2f}% |",
        f"| excluding 2008 | {err['excluding_bad_years']['weekdays']} | "
        f"{err['excluding_bad_years']['weekday_closures_missed']} | "
        f"{100 * err['excluding_bad_years']['error_rate']:.2f}% |",
        "",
        "That is 10–19 missed closures a year, so roughly 8 phantom rows across the "
        "forward block. The chosen rule was validated the same way: XBOM against "
        "2006H2–2026H1 disagrees on **18 of 6906 days (0.26%)** once 2008 is set aside "
        "— a twentyfold improvement, and the residue is almost entirely Muhurat and "
        "special Saturday sessions rather than ordinary holidays.",
        "",
        "Hardcoding only the fixed national dates (`01-26`, `04-14`, `05-01`, `08-15`, "
        "`10-02`, `12-25`) was measured too: those account for 114 of 397 historical "
        "closures (28.7%), leaving ~3.9% error. Every lunar festival — Diwali, Holi, "
        "Eid, Janmashtami — is still missed, so it buys little.",
        "",
        "Predicting closures from the astro features themselves was tested and "
        "**fails**: a gradient-boosted classifier over all 215 feature columns caught "
        "6 of 71 held-out holidays, worse than the 25 of 71 that bare calendar "
        "month/day features get on their own. NSE holidays track the civil festival "
        "calendar, not planetary position, so the index cannot be recovered from the "
        "data set.",
        "",
        "## Forward runway",
        "",
        "An **origin** is the last trading day of the past-60 encoder block. It is "
        f"usable when {spec['index']['future']} sessions exist strictly after it, since "
        "the known-covariate block spans them.",
        "",
        "| | Date |",
        "|-|------|",
        f"| First forward origin (last observed close) | `{run['first_forward_origin']}` |",
        f"| **Last origin with a full future-{spec['index']['future']} block** | "
        f"**`{run['last_origin_full_future_block']}`** |",
        f"| Last origin needing only the {spec['index']['horizon']}-step target horizon | "
        f"`{run['last_origin_horizon_only']}` |",
        f"| Forward origins available | {run['n_forward_origins']} |",
        "",
        f"The ephemeris ends {spec['source']['last_date']}, and that is the binding "
        "constraint — not data availability, since every feature is known-future (#3). "
        f"Forward inference therefore covers **{run['n_forward_origins']} origins**, "
        f"`{run['first_forward_origin']}` → `{run['last_origin_full_future_block']}`. "
        f"Origins between there and `{run['last_origin_horizon_only']}` are reachable "
        f"only if the decoder is shortened to {spec['index']['horizon']} steps — an open "
        "question in [#5](https://github.com/jenujari/prdict-pov-v1/issues/5).",
        "",
        "## What a mis-called date costs",
        "",
        "Nothing in training: the forward block has no target, so a wrong forward date "
        "cannot corrupt a fit. The damage is confined to inference, and it is **not** a "
        "mislabelled row — it is a shift.",
        "",
        "- **Phantom session** (predicted open, actually closed). The row enters the "
        "covariate block, so every later step in that window is fed the astro state of "
        "the wrong trading day and the returns come out shifted one step late. A window "
        "containing one phantom is wrong from the phantom onward, not just at it.",
        "- **Missing session** (predicted closed, actually open). The window skips a real "
        "day; predictions after it run one step early. Same shift, opposite sign. The "
        "unplaced Muhurat session is this case.",
        "",
        "Because the effect is a shift rather than a point error, there is no partial "
        "credit and no way to patch a single date after the fact — the fix is to correct "
        "`FORWARD_HOLIDAYS`, regenerate, and re-run inference from the affected origin. "
        "At a measured 0.26% that is expected to be zero or one date across the forward "
        "block, and NSE publishes its list far enough ahead that a correction lands "
        "before the date does.",
        "",
        "## Gaps in the observed history",
        "",
        f"Resolves [#25]({spec['gaps']['resolves']}). The observed index is authoritative "
        "for *which dates the market was open* — except where the source is simply missing "
        "rows. A missing session does not leave a hole: under map decision 1 the index "
        "collapses to trading days, so the two sessions either side become neighbours and "
        "the step between them silently spans more than one session's move.",
        "",
        "**Policy: the missing sessions are treated as ordinary trading holidays.** "
        "Nothing is filtered; all "
        f"{spec['gaps']['cost']['trainable_origins']} origins are trainable.",
        "",
        "Three facts make this the right default rather than a concession:",
        "",
        "1. **The step they produce already exists, in bulk.** 2008's missing sessions are "
        "mostly Fridays, so the surviving step is Thursday → Monday — the exact calendar "
        "spacing a genuine Friday holiday produces, of which the index holds hundreds. The "
        "target was never a uniform time interval: a normal weekend already makes one step "
        "span three calendar days against another's one.",
        "2. **The distortion is below the noise floor.** A straddling-return test — is the "
        "step across a gap inflated? — scores the confirmed 2008 gaps at mean z 0.67 "
        "against 0.24 for randomly chosen real holidays, with only 9 of 29 above z=1. The "
        "effect is not separable from ordinary holiday steps even knowing where to look.",
        "3. **Filtering costs the regime the hypothesis most needs.** 2008 is the sample's "
        "most extreme year (volatility rank 1 of 27, 45.8% annualised, 38 sessions beyond "
        "±4%). Excluding windows there trains a model that has never encoded those "
        "planetary configurations, and so cannot recognise them recurring.",
        "",
        "What is given up: "
        f"**{spec['gaps']['cost']['exposure']['origins_with_gap_in_target']} origins "
        f"({100 * spec['gaps']['cost']['exposure']['fraction']:.1f}%)** have one step of "
        "their 10-step target carrying roughly one extra session's move. Since the "
        "evaluation is a trading simulation driven by return magnitude, that exposure is "
        "recorded here rather than waved off — but it is one step in ten, in 2.3% of "
        "origins.",
        "",
        "The alternatives are kept in `kb/trading_calendar.json` so an ablation can "
        "reproduce them without re-deriving anything: filtering on the target span leaves "
        f"{spec['gaps']['cost']['if_filtered_on_target']['origins']} origins, and "
        "additionally requiring a clean encoder leaves "
        f"{spec['gaps']['cost']['if_filtered_on_whole_window']['origins']} — the latter "
        "retaining only 9 of 2008's 38 >4% sessions in any encoder block, against all 38 "
        "under the chosen policy. `cal.spans_gap(first, last)` applies either.",
        "",
        f"{spec['known_gaps']['note']}",
        "",
        "Two hypotheses were tested and rejected. **Crash closures**: the deficit starts in "
        "March and peaks in July, months before Lehman, and the crash sessions themselves "
        "(`2008-10-28` at −8.78%, `2008-10-31` at +6.99%) are all present — NSE used "
        "intraday circuit breakers, never whole-day closures. **A date shift**: shifting "
        "2008 by ±1 business day scores 0.863 against 0.859 unshifted, no improvement, "
        "where 2007 and 2009 score 0.992 and 1.000 unshifted.",
        "",
        "### How a gap is detected",
        "",
        "By **systematic source failure**, not date by date:",
        "",
        f"1. A year with at least **{spec['gaps']['detection']['year_threshold']}** XBOM "
        "mismatches. 2008 has 29; every other year has 0–2, so any threshold in 3–28 "
        "isolates it.",
        f"2. A run of at least **{spec['gaps']['detection']['run_length']}** consecutive "
        "closed weekdays, which no Indian holiday pattern produces. This is the only "
        "detector reaching before XBOM's 2006-08-03 start, and it fires exactly once, on "
        "`2002-08-27`–`2002-08-30`.",
        "",
        f"That gives **{spec['gaps']['cost']['n_gap_dates']} gap dates** and "
        f"**{spec['gaps']['cost']['n_discontinuities']} discontinuities** in the session "
        "index. They are identified so results can be interpreted and so an ablation can "
        "target them — not because anything is dropped.",
        "",
        "Note the gap dates are absent from the trading-day index entirely — no close means "
        "no row — so **their own** astro states are never encoded under any policy, "
        "including this one. Only their neighbours are. Encoding them would require a row "
        "with no target, which contradicts map decision 1.",
        "",
        "### Known residue, deliberately unflagged",
        "",
        f"{len(spec['gaps']['unflagged_mismatches'])} isolated XBOM mismatches sit outside "
        "the flagged years:",
        "",
        "```",
        "  ".join(spec["gaps"]["unflagged_mismatches"]),
        "```",
        "",
        "Each is individually indistinguishable from XBOM being wrong. A straddling-return "
        "test — does the step across the suspect date look inflated? — **cannot separate "
        "them from ordinary holidays, and cannot separate the confirmed 2008 gaps either** "
        "(mean z 0.67 for 2008 against 0.24 for random holidays). With no evidence to stand "
        "on, flagging all 11 would cost **16.2%** of trainable origins against 3.3% for the "
        "gaps above. Two of them are Saturdays, where the source consistently omits NSE's "
        "special sessions. Set `GAP_YEAR_THRESHOLD = 1` to include them.",
        "",
        "`2014-10-02`–`2014-10-06` was checked and is **genuine** — data and XBOM agree "
        "exactly (Gandhi Jayanti, Dussehra, Bakrid).",
        "",
        "### Consequence for the folds",
        "",
        "Since nothing is filtered, [#10](https://github.com/jenujari/prdict-pov-v1/issues/10) "
        "inherits **no** extra fold constraint — the session index is contiguous by "
        "construction and a gap is just another holiday. The one thing #10 should know is "
        "that 2008 carries the sample's densest run of them, so a fold boundary landing "
        "inside `2008-03-28`–`2008-11-28` sits in the least reliable stretch of the "
        "history. `cal.discontinuities` and `cal.spans_gap(first, last)` are available if "
        "the fold design wants to avoid it.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="re-check the frozen holiday list against exchange_calendars XBOM",
    )
    args = parser.parse_args()

    frame = pd.read_csv(CSV_PATH, usecols=[KEY, TARGET_SOURCE], parse_dates=[KEY])
    frame = frame.sort_values(KEY).reset_index(drop=True)
    spec = build(frame)

    CAL_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    CAL_MD.write_text(render_markdown(spec))

    print(f"history sessions : {spec['history']['n_sessions']}")
    print(f"forward sessions : {spec['forward']['n_sessions']}")
    print(f"total index      : {spec['index']['n_sessions']}")
    print(f"weekday-only error: {100 * spec['weekday_rule_error']['all_history']['error_rate']:.2f}%"
          f" (excl 2008: {100 * spec['weekday_rule_error']['excluding_bad_years']['error_rate']:.2f}%)")
    print(f"last usable origin: {spec['runway']['last_origin_full_future_block']}"
          f"  ({spec['runway']['n_forward_origins']} forward origins)")
    cost = spec["gaps"]["cost"]
    print(f"gaps             : {cost['n_gap_dates']} dates, "
          f"{cost['n_discontinuities']} discontinuities")
    print(f"trainable origins: {cost['trainable_origins']} (gaps kept as holidays)")
    print(f"  of which a gap falls inside the target: "
          f"{cost['exposure']['origins_with_gap_in_target']} "
          f"({100 * cost['exposure']['fraction']:.1f}%)")

    if args.audit:
        audit(spec)

    print(f"\nwrote {CAL_JSON.relative_to(ROOT)} and {CAL_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
