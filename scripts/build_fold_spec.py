"""Derive the walk-forward fold, holdout, purge and inner-split spec.

Resolves wayfinder ticket #10.
Writes kb/fold_spec.json (machine-readable) and kb/fold_spec.md (human-readable).

Every boundary here is *derived* from the trading calendar and three stated
choices — rolling geometry, ten training years, five folds. Nothing is a
hand-picked date. The build fails if a derived boundary stops satisfying the
separation the purge is supposed to buy.

Run inside container:
    ./container/run.sh python scripts/build_fold_spec.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from prdict.trading_calendar import TradingCalendar, load_calendar

ROOT = Path(__file__).resolve().parent.parent
SPEC_JSON = ROOT / "kb" / "fold_spec.json"
SPEC_MD = ROOT / "kb" / "fold_spec.md"

# --- the choices everything else is derived from -----------------------------

GEOMETRY = "rolling_window"
N_FOLDS = 5
TRAIN_YEARS = 10

# Purge only. There is no embargo: see EMBARGO_OMITTED.
PURGE_SESSIONS = 10

# The holdout opens on the first session of the calendar year after the last
# validation fold. It is a boundary, not a tuned length.
HOLDOUT_START = "2024-01-01"

# Fraction of each training block reserved as the inner validation set that
# early stopping and hyperparameter search are allowed to see.
INNER_VAL_FRACTION = 0.15

EMBARGO_OMITTED = (
    "An embargo withholds training rows that sit *after* a validation block. Under any "
    "forward walk-forward geometry — rolling or expanding — every training block ends "
    "before its own validation block begins, so there are no such rows to withhold. An "
    "embargo only does work in interleaved k-fold CV, where a test block has training "
    "data on both sides. The previous revision of this spec recorded a 69-session "
    "embargo; it withheld nothing, and all 69 of fold 1's nominally embargoed origins "
    "were present in the training sets of folds 3, 4 and 5."
)

IN_FOLD_FITS = [
    {
        "stage": "Scaler",
        "description": (
            "StandardScaler / RobustScaler, fitted on the inner training block only, so "
            "that the inner validation set is scaled by statistics that never saw it."
        ),
    },
    {
        "stage": "Correlation prune",
        "description": (
            "Pairwise \\|r\\| >= 0.95 among features (map decision 7, stage 1). Refit per "
            "fold, and bound by the pair-atomicity constraint below."
        ),
    },
    {
        "stage": "Feature ranking",
        "description": (
            "Mutual information + Spearman against the target, keep top-N (map decision "
            "7, stage 2). Target-dependent, so it must never see validation labels."
        ),
    },
    {
        "stage": "PCA",
        "description": (
            "Per-timestep, numeric-only, n_components=0.95 (map decision 9), fitted on "
            "the scaled inner training block."
        ),
    },
    {
        "stage": "Empirical variance / redundancy filters",
        "description": (
            "Any filter whose decision depends on values observed in this fold — "
            "near-zero variance, empirical duplicate detection. Distinct from the "
            "structural drops below."
        ),
    },
]

GLOBAL_FITS = [
    {
        "stage": "Structural constant drop",
        "description": (
            "The 28 constant columns from #3 are dropped once, globally. "
            "`kb/column_spec.md` records that all 28 are constant *by definition* — a "
            "formula constant for the nodes, a mean-node rate, and so on — not constant "
            "by sampling accident. A definitional constant carries no fold-specific "
            "information, so dropping it globally cannot leak. This is the concern #10 "
            "originally raised, and #3 answers it."
        ),
    },
    {
        "stage": "Structural Ketu drops",
        "description": (
            "`ketu_longitude` and the duplicate Ketu balas (#7, #30) are dropped "
            "globally on the same argument: they are exact functions of their Rahu twins "
            "on every row, definitionally rather than empirically."
        ),
    },
    {
        "stage": "Categorical level list",
        "description": (
            "Levels come from `categories_list.json` — the full declared list, never the "
            "observed values (map decision 6). Fitting an encoder per fold would give "
            "different embedding widths per fold; the declared list makes the encoding "
            "fold-invariant by construction."
        ),
    },
    {
        "stage": "Angular sin/cos transform",
        "description": (
            "Deterministic and parameter-free (#7). Nothing is estimated, so there is "
            "nothing to fit inside a fold."
        ),
    },
]

CONSTRAINTS = [
    {
        "name": "Sin/cos pair atomicity",
        "rule": (
            "The correlation prune and the ranking stage treat each `<col>_sin` / "
            "`<col>_cos` pair as one indivisible unit: both survive or both are dropped, "
            "and the pair is ranked on its joint score."
        ),
        "why": (
            "#7 settled that a lone half is a wrong encoding, not merely a weak one — "
            "sin alone folds two configurations onto one value. Because the prune is "
            "refit per fold, an unconstrained prune could keep a pair in fold 1 and "
            "split it in fold 4, giving the concatenated out-of-sample series several "
            "different feature semantics. #7 binds #12 and #13 to this; the same binding "
            "applies here, because the in-fold prune is the same operation."
        ),
    },
    {
        "name": "No target transform is fitted",
        "rule": "The 10-step log-return target passes through untransformed in every fold.",
        "why": (
            "#8 rejected winsorisation — the extreme days are the P&L under map decision "
            "8 — so no target statistic is estimated anywhere and the target cannot leak "
            "across a fold edge."
        ),
    },
]


def sessions_per_year(cal: TradingCalendar) -> int:
    """Median session count over complete observed years.

    Converts the stated ten-year training requirement into a session count, so
    the training length is derived rather than picked as a date.
    """
    counts = pd.Series(cal.history.year).value_counts().sort_index()
    complete = counts.iloc[1:-1]  # first and last observed years are partial
    return int(complete.median())


def inner_split(cal: TradingCalendar, train: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """Split a training block into inner-train and inner-validation, purged."""
    val = train[-round(len(train) * INNER_VAL_FRACTION) :]
    cutoff = cal.position(val[0]) - PURGE_SESSIONS
    return {"train": train[pd.Index(train.map(cal.position)) <= cutoff], "val": val}


def build() -> dict:
    cal = load_calendar()
    origins = cal.trainable_origins()

    holdout_start = pd.Timestamp(HOLDOUT_START)
    cv = origins[origins < holdout_start]
    holdout = origins[origins >= holdout_start]
    cv_pos = pd.Index(cv.map(cal.position))

    spy = sessions_per_year(cal)
    train_len = TRAIN_YEARS * spy
    val_pool = len(cv) - train_len - PURGE_SESSIONS
    val_len = val_pool // N_FOLDS

    if val_len < 1:
        raise ValueError(
            f"{TRAIN_YEARS} training years ({train_len} origins) plus a {PURGE_SESSIONS}-session "
            f"purge leaves {val_pool} origins for {N_FOLDS} folds — shorten the training block "
            f"or cut folds. The CV pool holds {len(cv)} origins."
        )
    if round(train_len * INNER_VAL_FRACTION) < 1:
        raise ValueError(
            f"INNER_VAL_FRACTION={INNER_VAL_FRACTION} gives an empty inner validation set; "
            "early stopping would have nothing to watch."
        )
    remainder = val_pool - val_len * N_FOLDS
    first_val_i = len(cv) - val_pool

    folds = []
    for i in range(N_FOLDS):
        lo = first_val_i + i * val_len
        hi = len(cv) - 1 if i == N_FOLDS - 1 else first_val_i + (i + 1) * val_len - 1
        val = cv[lo : hi + 1]

        train = cv[cv_pos <= cal.position(val[0]) - PURGE_SESSIONS][-train_len:]
        inner = inner_split(cal, train)

        folds.append(
            {
                "fold": i + 1,
                "train_start": str(train[0].date()),
                "train_end": str(train[-1].date()),
                "train_origins": len(train),
                "inner_train_start": str(inner["train"][0].date()),
                "inner_train_end": str(inner["train"][-1].date()),
                "inner_train_origins": len(inner["train"]),
                "inner_val_start": str(inner["val"][0].date()),
                "inner_val_end": str(inner["val"][-1].date()),
                "inner_val_origins": len(inner["val"]),
                "val_start": str(val[0].date()),
                "val_end": str(val[-1].date()),
                "val_origins": len(val),
                "train_val_separation_sessions": cal.position(val[0]) - cal.position(train[-1]),
                "train_spans_gap": cal.spans_gap(cal.position(train[0]), cal.position(train[-1])),
                "val_spans_gap": cal.spans_gap(cal.position(val[0]), cal.position(val[-1])),
            }
        )

    final_train = cv[cv_pos <= cal.position(holdout[0]) - PURGE_SESSIONS][-train_len:]
    final_inner = inner_split(cal, final_train)
    n_val = sum(f["val_origins"] for f in folds)

    spec = {
        "generated": str(date.today()),
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/10",
        "geometry": GEOMETRY,
        "total_trainable_origins": len(origins),
        "derivation": {
            "sessions_per_year": spy,
            "train_years": TRAIN_YEARS,
            "train_length_origins": train_len,
            "n_folds": N_FOLDS,
            "val_length_origins": val_len,
            "last_fold_extra_origins": remainder,
            "cv_pool_origins": len(cv),
            "identity": (
                f"{len(cv)} CV origins = {train_len} train + {PURGE_SESSIONS} purge + "
                f"{N_FOLDS} x {val_len} validation + {remainder} remainder"
            ),
        },
        "purge": {
            "sessions": PURGE_SESSIONS,
            "derivation": (
                "A sample at origin t carries labels for the closes at t+1..t+10, so two "
                "origins separated by fewer than 10 sessions share at least one label day "
                "(#8). Separation of exactly 10 is therefore the floor at which no "
                "training label and no validation label share a return. This spec sits on "
                "that floor: each fold's last training label terminates on the close of "
                "its first validation origin — the same close, but never the same return."
            ),
            "known_residual": (
                "At 10 sessions the separation is label-level, not row-level. Each "
                "training block's 30-session known-future block still reaches 21 sessions "
                "into its validation dates, so the scaler and PCA are fitted on rows "
                "carrying validation-period dates. Accepted deliberately: every one of "
                "those columns is astro or calendar (#3 — every feature is known-future), "
                "so the rows hold no information that would not have been available at "
                "the training origin. Row-level separation would cost 69 sessions "
                "(60 encoder + 9 label) and 59 training origins per fold. Recorded so the "
                "choice is visible rather than implied."
            ),
        },
        "embargo": {"sessions": 0, "why_omitted": EMBARGO_OMITTED},
        "holdout": {
            "start_date": str(holdout[0].date()),
            "end_date": str(holdout[-1].date()),
            "origins": len(holdout),
            "purge_sessions": PURGE_SESSIONS,
            "final_train_start": str(final_train[0].date()),
            "final_train_end": str(final_train[-1].date()),
            "final_train_origins": len(final_train),
            "final_inner_val_start": str(final_inner["val"][0].date()),
            "final_inner_val_end": str(final_inner["val"][-1].date()),
            "policy": (
                "Untouched until all three models are final and compared once. The models "
                "scored on it are refit on the final training block above — the same "
                "rolling geometry, the same purge — so the holdout comparison is not a "
                "different experiment from the folds."
            ),
            "known_limitation": (
                "The window is a calm rising regime: +9.8% total, 13.9% annualised "
                "volatility, worst drawdown -15.8%, worst single day -6.1%. The history "
                "it is drawn from contains far harsher regimes (2008: -51.8%, 45.8% vol, "
                "-59.4% drawdown; 2020: -38.4% drawdown, -13.9% worst day). Because the "
                "always-up baseline scores well on a rising window by construction, this "
                "holdout has limited power to separate the astro models from that "
                "baseline on drawdown or Sharpe. Chosen deliberately to keep the CV pool "
                "large; the write-up must state that the comparison is evidenced on one "
                "benign regime only."
            ),
        },
        "walk_forward": {"n_folds": N_FOLDS, "total_val_origins": n_val, "folds": folds},
        "inner_split": {
            "fraction": INNER_VAL_FRACTION,
            "purge_sessions": PURGE_SESSIONS,
            "rule": (
                f"The last {int(INNER_VAL_FRACTION * 100)}% of each training block is the "
                "inner validation set. The inner training block is everything before it, "
                "less the same 10-session purge. Early stopping and hyperparameter "
                "selection see the inner validation set and nothing else."
            ),
            "why": (
                "The outer validation origins carry the reported out-of-sample trading "
                "scorecard. If early stopping or hyperparameter choice also read them, "
                "the reported number becomes a best-of-N selected on the very days it is "
                "reported for. Overlapping labels make this acute: 9 of every 10 label "
                f"days are shared, so a fold's validation block is worth roughly "
                f"{folds[0]['val_origins'] // 10} independent observations, and best-of-N "
                "on that few will manufacture an edge out of noise."
            ),
            "budget_rule": (
                "The hyperparameter search space and the number of configurations tried "
                "are fixed in writing before any model runs, and are not expanded after "
                "seeing inner validation results."
            ),
        },
        "in_fold_fits": IN_FOLD_FITS,
        "global_fits": GLOBAL_FITS,
        "constraints": CONSTRAINTS,
        "gap_audit": gap_audit(cal, folds),
        "fold_reporting": {
            "headline": (
                "One concatenated out-of-sample series. The five validation blocks are "
                "contiguous and disjoint, so their predictions join into a single "
                f"{n_val}-origin series spanning {folds[0]['val_start']} to "
                f"{folds[-1]['val_end']}, scored once on the trading-simulation scorecard "
                "(map decision 8). This is the headline number."
            ),
            "secondary": (
                "Per-fold metrics reported as mean +- std across the five folds, as a "
                "consistency check on the headline. A headline carried by one fold is "
                "reported as such."
            ),
            "fold_boundary_rule": (
                "Each fold is a separately fitted model, so the concatenated series "
                "crosses four model changes. Positions are flattened at each fold "
                "boundary and reopened by the incoming model, and the transaction costs "
                "of doing so are charged. Carrying a position across a model change would "
                "attribute one model's entry to another model's exit."
            ),
            "effective_sample_size": (
                f"Consecutive origins share 9 of 10 label days, so the {n_val}-origin "
                f"series is worth roughly {n_val // 10} independent observations. This "
                "does not change the purge — the purge is fixed by label overlap at the "
                "fold edge, not by dependence within a block — but it governs "
                "interpretation: confidence intervals and any significance test must use "
                "the effective count, not the origin count."
            ),
        },
    }

    validate(cal, spec, cv, holdout)
    return spec


def gap_audit(cal: TradingCalendar, folds: list[dict]) -> dict:
    """Where #25's 2008 discontinuities land relative to the fold boundaries."""
    disc = sorted(cal.discontinuities)
    boundaries = {
        f"fold{f['fold']}.{key}": cal.position(pd.Timestamp(f[key]))
        for f in folds
        for key in ("train_start", "train_end", "val_start", "val_end")
    }
    near = {
        name: [str(cal.sessions[p].date()) for p in disc if abs(p - pos) <= 69]
        for name, pos in boundaries.items()
    }
    near = {k: v for k, v in near.items() if v}

    clean = not any(f["val_spans_gap"] for f in folds)
    scoring_near = {k: v for k, v in near.items() if k.split(".")[1].startswith("val")}
    return {
        "n_discontinuities": len(disc),
        "source": "2008's 21 missing sessions (#25), treated as ordinary trading holidays.",
        "boundaries_within_69_sessions_of_a_gap": near,
        "n_validation_boundaries_near_a_gap": len(scoring_near),
        "n_training_boundaries_near_a_gap": len(near) - len(scoring_near),
        "verdict": (
            "No validation block contains or comes within 69 sessions of a discontinuity, "
            "so no scored origin is affected and no fold boundary needs to move. #25's "
            "policy stands: nothing is filtered."
            if clean and not scoring_near
            else "A validation boundary sits on or near a known gap — revisit #25's policy "
            "for that fold."
        ),
        "training_boundaries_note": (
            "Only training-block edges land near the 2008 stretch, and only because the "
            "rolling length places them there — no boundary was chosen to avoid or hit a "
            "gap. Fold 4's training block opens inside the defective run, so it carries a "
            "partial 2008. Under #25 that is not a defect to correct: the missing sessions "
            "are ordinary trading holidays, and a training block that starts mid-year is "
            "no more unusual than one that starts mid-quarter."
        ),
        "rolling_cost": (
            f"Under the rolling geometry fold 5 trains from {folds[-1]['train_start']} and "
            f"therefore never sees 2008 or 2000-01; fold 4 picks 2008 up only from "
            f"{folds[3]['train_start']}. Combined with a holdout drawn from a calm regime, "
            "the later folds and the final comparison are trained and scored on "
            "progressively calmer data. This is the price of holding the training size "
            "constant, and it is recorded rather than hidden."
        ),
    }


def validate(
    cal: TradingCalendar, spec: dict, cv: pd.DatetimeIndex, holdout: pd.DatetimeIndex
) -> None:
    """Assert the properties the spec claims. The build fails if one stops holding."""
    folds = spec["walk_forward"]["folds"]
    purge = spec["purge"]["sessions"]
    d = spec["derivation"]

    assert len(folds) == N_FOLDS, f"expected {N_FOLDS} folds, got {len(folds)}"

    lengths = {f["train_origins"] for f in folds}
    assert lengths == {d["train_length_origins"]}, (
        f"rolling geometry requires one training length, got {sorted(lengths)}"
    )

    for f in folds:
        assert f["train_val_separation_sessions"] >= purge, (
            f"fold {f['fold']}: separation {f['train_val_separation_sessions']} < purge {purge}"
        )
        # No training origin's label window may reach into its own validation block.
        last_train_label = cal.position(pd.Timestamp(f["train_end"])) + cal.horizon
        first_val_label = cal.position(pd.Timestamp(f["val_start"])) + 1
        assert last_train_label < first_val_label, (
            f"fold {f['fold']}: training labels reach session {last_train_label}, "
            f"validation labels open at {first_val_label}"
        )
        # Inner validation is disjoint from, and precedes, outer validation.
        assert pd.Timestamp(f["inner_val_end"]) <= pd.Timestamp(f["train_end"])
        assert pd.Timestamp(f["inner_val_start"]) > pd.Timestamp(f["inner_train_end"])
        inner_sep = cal.position(pd.Timestamp(f["inner_val_start"])) - cal.position(
            pd.Timestamp(f["inner_train_end"])
        )
        assert inner_sep >= purge, f"fold {f['fold']}: inner separation {inner_sep} < {purge}"
        assert f["inner_train_origins"] + f["inner_val_origins"] <= f["train_origins"]

    # Validation blocks are contiguous, disjoint, and exhaust the CV pool's tail.
    for a, b in zip(folds, folds[1:]):
        gap = cal.position(pd.Timestamp(b["val_start"])) - cal.position(pd.Timestamp(a["val_end"]))
        assert gap == 1, f"validation blocks {a['fold']}/{b['fold']} not contiguous (gap {gap})"
    assert pd.Timestamp(folds[-1]["val_end"]) == cv[-1], "last fold must end at the CV pool's end"

    # The holdout is absent from every training and validation block.
    for f in folds:
        assert pd.Timestamp(f["val_end"]) < holdout[0]
        assert pd.Timestamp(f["train_end"]) < holdout[0]

    # The final training block is purged from the holdout by the same rule.
    final_end = pd.Timestamp(spec["holdout"]["final_train_end"])
    final_sep = cal.position(holdout[0]) - cal.position(final_end)
    assert final_sep >= purge, f"final train/holdout separation {final_sep} < purge {purge}"
    assert cal.position(final_end) + cal.horizon < cal.position(holdout[0]) + 1, (
        "final training labels reach into the holdout"
    )

    # The counting identity, with the purged origins named rather than absorbed.
    assert (
        d["train_length_origins"]
        + purge
        + d["n_folds"] * d["val_length_origins"]
        + d["last_fold_extra_origins"]
        == d["cv_pool_origins"]
    ), "CV pool does not decompose into train + purge + validation blocks"
    assert d["cv_pool_origins"] + spec["holdout"]["origins"] == spec["total_trainable_origins"]

    assert spec["embargo"]["sessions"] == 0, "an embargo is vacuous under forward walk-forward"


def render_markdown(spec: dict) -> str:
    d = spec["derivation"]
    folds = spec["walk_forward"]["folds"]
    ho = spec["holdout"]
    ga = spec["gap_audit"]

    lines = [
        "# Walk-forward folds, holdout, and purge arithmetic",
        "",
        f"Generated {spec['generated']} by `scripts/build_fold_spec.py`. Do not hand-edit.",
        f"Resolves [#10]({spec['resolves']}).",
        "",
        "## Summary",
        "",
        f"- **Geometry**: rolling window — every fold trains on exactly "
        f"`{d['train_length_origins']}` origins (~{TRAIN_YEARS} trading years).",
        f"- **Folds**: {d['n_folds']}, validating `{spec['walk_forward']['total_val_origins']}` "
        f"origins from `{folds[0]['val_start']}` to `{folds[-1]['val_end']}`.",
        f"- **Purge**: `{spec['purge']['sessions']}` sessions before every validation block. "
        "No embargo.",
        f"- **Holdout**: `{ho['origins']}` origins, `{ho['start_date']}` → `{ho['end_date']}`.",
        f"- **Total trainable origins**: `{spec['total_trainable_origins']}`.",
        "",
        "## 1. Why rolling, not expanding",
        "",
        "Under an expanding window the last fold trains on roughly twice the data of the",
        "first. A fold-to-fold difference in score then confounds two causes — the regime",
        "changed, or the model simply had more data — and the concatenated series is",
        "produced by five models of systematically increasing capability. Holding the",
        "training length fixed removes that confound, so fold-to-fold differences are about",
        "regime alone. The cost is recorded in §7: the later folds drop the early history,",
        "so fold 5 never sees 2008.",
        "",
        "## 2. Derivation",
        "",
        "Nothing below is a hand-picked date. Three choices are stated; every boundary",
        "follows from them and the trading calendar.",
        "",
        "| Quantity | Value | Source |",
        "|---|---|---|",
        f"| Sessions per trading year | {d['sessions_per_year']} | median over complete "
        "observed years |",
        f"| Training years | {d['train_years']} | **chosen** — long enough to span the "
        "2000-01 bear, the 2003-07 bull and a crash |",
        f"| Training length | {d['train_length_origins']} origins | derived: "
        f"{d['train_years']} x {d['sessions_per_year']} |",
        f"| Folds | {d['n_folds']} | **chosen** — see the effective sample size in §6 |",
        f"| Validation per fold | {d['val_length_origins']} origins | derived from the "
        "remainder |",
        f"| CV pool | {d['cv_pool_origins']} origins | trainable origins before the holdout |",
        "",
        f"`{d['identity']}`",
        "",
        "## 3. Folds",
        "",
        "| Fold | Inner train | Inner val | Validation |",
        "|---|---|---|---|",
    ]

    for f in folds:
        lines.append(
            f"| {f['fold']} | `{f['inner_train_start']}` → `{f['inner_train_end']}` "
            f"({f['inner_train_origins']}) | `{f['inner_val_start']}` → `{f['inner_val_end']}` "
            f"({f['inner_val_origins']}) | `{f['val_start']}` → `{f['val_end']}` "
            f"({f['val_origins']}) |"
        )

    lines += [
        "",
        f"Inner train plus inner val is the whole training block for that fold "
        f"(`{d['train_length_origins']}` origins, less the inner purge). Every fold's "
        f"train-to-validation separation is exactly "
        f"`{folds[0]['train_val_separation_sessions']}` sessions.",
        "",
        "## 4. Purge",
        "",
        f"**{spec['purge']['sessions']} sessions, before every validation block.**",
        "",
        spec["purge"]["derivation"],
        "",
        "**Known residual.** " + spec["purge"]["known_residual"],
        "",
        "**No embargo.** " + spec["embargo"]["why_omitted"],
        "",
        "## 5. Holdout",
        "",
        f"- **Range**: `{ho['start_date']}` → `{ho['end_date']}` ({ho['origins']} origins).",
        f"- **Final training block**: `{ho['final_train_start']}` → `{ho['final_train_end']}` "
        f"({ho['final_train_origins']} origins), purged by `{ho['purge_sessions']}` sessions "
        "from the first holdout origin — the same rolling geometry and the same purge as "
        "every fold.",
        f"- **Final inner validation**: `{ho['final_inner_val_start']}` → "
        f"`{ho['final_inner_val_end']}`.",
        f"- **Policy**: {ho['policy']}",
        "",
        "**Known limitation.** " + ho["known_limitation"],
        "",
        "## 6. Reporting",
        "",
        f"- **Headline**: {spec['fold_reporting']['headline']}",
        f"- **Secondary**: {spec['fold_reporting']['secondary']}",
        f"- **Fold boundaries**: {spec['fold_reporting']['fold_boundary_rule']}",
        f"- **Effective sample size**: {spec['fold_reporting']['effective_sample_size']}",
        "",
        "### Model selection",
        "",
        spec["inner_split"]["why"],
        "",
        f"**Rule.** {spec['inner_split']['rule']}",
        "",
        f"**Budget.** {spec['inner_split']['budget_rule']}",
        "",
        "## 7. Gap audit (#25)",
        "",
        f"- Discontinuities on the index: **{ga['n_discontinuities']}** — {ga['source']}",
        f"- Validation boundaries within 69 sessions of one: "
        f"**{ga['n_validation_boundaries_near_a_gap'] or 'none'}**",
        f"- Training boundaries within 69 sessions of one: "
        f"**{ga['n_training_boundaries_near_a_gap'] or 'none'}**",
        f"- {ga['verdict']}",
        "",
        ga["training_boundaries_note"],
        "",
        ga["rolling_cost"],
        "",
        "## 8. What is fitted where",
        "",
        "### Fitted inside every fold, on the inner training block only",
        "",
        "| Stage | Detail |",
        "|---|---|",
    ]

    for fit in spec["in_fold_fits"]:
        lines.append(f"| **{fit['stage']}** | {fit['description']} |")

    lines += [
        "",
        "### Applied once, globally — and why that cannot leak",
        "",
        "| Stage | Detail |",
        "|---|---|",
    ]

    for fit in spec["global_fits"]:
        lines.append(f"| **{fit['stage']}** | {fit['description']} |")

    lines += [
        "",
        "The distinction is *structural versus empirical*, not *cheap versus expensive*. A",
        "column dropped because a formula makes it constant carries no fold-specific",
        "information and may be dropped once. A column dropped because it happened to look",
        "constant in this sample must be re-decided in every fold.",
        "",
        "### Constraints binding the in-fold stages",
        "",
    ]

    for c in spec["constraints"]:
        lines += [f"**{c['name']}.** {c['rule']}", "", f"*Why:* {c['why']}", ""]

    return "\n".join(lines)


def main() -> None:
    spec = build()
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    SPEC_MD.write_text(render_markdown(spec))
    print(f"Wrote {SPEC_JSON.relative_to(ROOT)} and {SPEC_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
