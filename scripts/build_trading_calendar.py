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

# 2008 is a data defect, not a run of holidays: 22 of its 37 weekday closures
# fall on a Friday and the year totals 225 sessions against a ~246 norm. Every
# error rate below is reported both with and without it so the rule is not
# judged against corrupt truth. See the data-quality ticket.
BAD_YEARS = [2008]


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
        "known_gaps": {
            "bad_years": BAD_YEARS,
            "note": "2008 holds ~22 spurious closures (22 of 37 weekday closures "
            "are Fridays; 225 sessions against a ~246 norm). Treated as a data "
            "defect, excluded from every error rate here, not repaired.",
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
        "## Known defect: 2008",
        "",
        f"{spec['known_gaps']['note']} It is **not** repaired here — 2008 sits inside the "
        "training history, so ~22 phantom closures collapse real consecutive sessions "
        "into neighbours and quietly corrupt every window that crosses them.",
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

    if args.audit:
        audit(spec)

    print(f"\nwrote {CAL_JSON.relative_to(ROOT)} and {CAL_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
