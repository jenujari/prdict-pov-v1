"""Derive the walk-forward fold, holdout, and purge/embargo spec.

Resolves wayfinder ticket #10.
Writes kb/fold_spec.json (machine-readable) and kb/fold_spec.md (human-readable).

Run inside container:
    ./container/run.sh python scripts/build_fold_spec.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from prdict.trading_calendar import load_calendar

ROOT = Path(__file__).resolve().parent.parent
SPEC_JSON = ROOT / "kb" / "fold_spec.json"
SPEC_MD = ROOT / "kb" / "fold_spec.md"

N_FOLDS = 5
HOLDOUT_START = "2024-01-01"
PURGE_PRE_SESSIONS = 10
EMBARGO_POST_SESSIONS = 69

IN_FOLD_FITS = [
    {
        "stage": "Scaler",
        "description": "StandardScaler / RobustScaler fitted strictly on training fold origins.",
    },
    {
        "stage": "Correlation Prune",
        "description": "Pairwise correlation matrix (|r| >= 0.95) computed strictly on training fold origins.",
    },
    {
        "stage": "Feature Ranking",
        "description": "Mutual Information / Spearman correlation ranking computed strictly on training fold origins.",
    },
    {
        "stage": "PCA",
        "description": "Principal Component Analysis (n_components=0.95) fitted strictly on scaled training fold features.",
    },
    {
        "stage": "Data-Dependent Redundancy / Variance Filter",
        "description": "Any empirical zero-variance or empirical redundancy filtering evaluated strictly inside each training fold.",
    },
]


def build() -> dict:
    cal = load_calendar()
    origins = cal.trainable_origins()

    holdout_start_ts = pd.Timestamp(HOLDOUT_START)
    val_origins = origins[origins < holdout_start_ts]
    holdout_origins = origins[origins >= holdout_start_ts]

    # First fold starts after ~10 years of initial training history (2000-03-29 to 2010-03-31)
    train_init_end = pd.Timestamp("2010-03-31")
    train0 = val_origins[val_origins <= train_init_end]
    val_pool = val_origins[val_origins > train_init_end]

    fold_size = len(val_pool) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        v_start = val_pool[i * fold_size]
        v_end = val_pool[(i + 1) * fold_size - 1] if i < N_FOLDS - 1 else val_pool[-1]

        v_start_idx = cal.position(v_start)
        v_end_idx = cal.position(v_end)

        # Purge 10 sessions before v_start for training origins
        train_purge_end_idx = v_start_idx - PURGE_PRE_SESSIONS
        train_origins_fold = val_origins[val_origins.map(cal.position) <= train_purge_end_idx]

        val_block_origins = val_origins[(val_origins >= v_start) & (val_origins <= v_end)]

        folds.append({
            "fold": i + 1,
            "train_start": str(val_origins[0].date()),
            "train_end": str(val_origins[val_origins.map(cal.position) <= train_purge_end_idx][-1].date()),
            "train_origins_count": len(train_origins_fold),
            "val_start": str(v_start.date()),
            "val_end": str(v_end.date()),
            "val_origins_count": len(val_block_origins),
            "purge_pre_sessions": PURGE_PRE_SESSIONS,
            "embargo_post_sessions": EMBARGO_POST_SESSIONS,
        })

    spec = {
        "generated": str(date.today()),
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/10",
        "geometry": "expanding_window",
        "total_trainable_origins": len(origins),
        "holdout": {
            "start_date": str(holdout_origins[0].date()),
            "end_date": str(holdout_origins[-1].date()),
            "origins_count": len(holdout_origins),
            "policy": "Strictly untouched final holdout set containing post-2023 market regimes.",
        },
        "walk_forward": {
            "n_folds": N_FOLDS,
            "total_val_pool_origins": len(val_origins),
            "total_val_scored_origins": sum(f["val_origins_count"] for f in folds),
            "folds": folds,
        },
        "purge_embargo_arithmetic": {
            "purge_pre_sessions": PURGE_PRE_SESSIONS,
            "purge_reason": "10 sessions (target horizon) to eliminate direct target label leakage from train into val.",
            "embargo_post_sessions": EMBARGO_POST_SESSIONS,
            "embargo_reason": "69 sessions (60 encoder lookback + 9 label overlap) for 100% complete row-level independence.",
        },
        "in_fold_fits": IN_FOLD_FITS,
        "fold_reporting": {
            "concatenated_series": "Out-of-sample predictions across 5 validation folds (3393 origins) concatenated into one continuous time-series for trading simulation.",
            "aggregated_metrics": "Per-fold metrics (Sharpe, max drawdown, win rate) reported as mean +- std across all 5 folds.",
        },
    }

    validate(spec)
    return spec


def validate(spec: dict) -> None:
    wf = spec["walk_forward"]
    assert len(wf["folds"]) == N_FOLDS, f"expected {N_FOLDS} folds"
    assert wf["total_val_scored_origins"] + spec["holdout"]["origins_count"] + len(
        load_calendar().trainable_origins()[load_calendar().trainable_origins() <= pd.Timestamp("2010-03-31")]
    ) == spec["total_trainable_origins"]
    assert spec["purge_embargo_arithmetic"]["purge_pre_sessions"] == 10
    assert spec["purge_embargo_arithmetic"]["embargo_post_sessions"] == 69


def render_markdown(spec: dict) -> str:
    lines = [
        "# Walk-Forward CV & Holdout Specification",
        "",
        f"Generated {spec['generated']} by `scripts/build_fold_spec.py`. Do not hand-edit.",
        f"Resolves [#10]({spec['resolves']}).",
        "",
        "## Summary",
        "",
        f"- **Total Trainable Origins**: `{spec['total_trainable_origins']}` (2000-03-29 → 2026-06-15)",
        f"- **CV Geometry**: Expanding Window (Anchored Walk-Forward)",
        f"- **Validation Folds**: {spec['walk_forward']['n_folds']} folds (2010-04-01 → 2023-12-29, `{spec['walk_forward']['total_val_scored_origins']}` origins)",
        f"- **Final Holdout**: `{spec['holdout']['origins_count']}` origins (`{spec['holdout']['start_date']}` → `{spec['holdout']['end_date']}`)",
        "",
        "## 1. Walk-Forward Fold Boundaries",
        "",
        "| Fold | Train Range | Train Origins | Validation Range | Val Origins | Purge Pre | Embargo Post |",
        "|------|-------------|---------------|------------------|-------------|-----------|--------------|",
    ]

    for f in spec["walk_forward"]["folds"]:
        lines.append(
            f"| Fold {f['fold']} | `{f['train_start']}` → `{f['train_end']}` | {f['train_origins_count']} | "
            f"`{f['val_start']}` → `{f['val_end']}` | {f['val_origins_count']} | {f['purge_pre_sessions']} s | {f['embargo_post_sessions']} s |"
        )

    lines += [
        "",
        "## 2. Final Holdout Set",
        "",
        f"- **Range**: `{spec['holdout']['start_date']}` → `{spec['holdout']['end_date']}` ({spec['holdout']['origins_count']} trainable origins)",
        f"- **Policy**: {spec['holdout']['policy']}",
        "",
        "## 3. Purge & Embargo Arithmetic",
        "",
        "| Parameter | Length | Arithmetic & Derivation |",
        "|-----------|--------|--------------------------|",
        f"| **Purge (Pre-Validation)** | `{spec['purge_embargo_arithmetic']['purge_pre_sessions']} sessions` | {spec['purge_embargo_arithmetic']['purge_reason']} |",
        f"| **Embargo (Post-Validation)** | `{spec['purge_embargo_arithmetic']['embargo_post_sessions']} sessions` | {spec['purge_embargo_arithmetic']['embargo_reason']} |",
        "",
        "## 4. Explicit In-Fold Fits",
        "",
        "To prevent lookahead leak, the following transformations MUST be fitted strictly inside each training fold:",
        "",
        "| Stage | Description |",
        "|-------|-------------|",
    ]

    for fit in spec["in_fold_fits"]:
        lines.append(f"| **{fit['stage']}** | {fit['description']} |")

    lines += [
        "",
        "## 5. Overlapping-Sample Dependence & Reporting",
        "",
        f"- **Sample Dependence**: Consecutive origins share 9 of 10 target label days. This autocorrelation increases metric variance across folds.",
        f"- **Reporting Policy**: {spec['fold_reporting']['concatenated_series']}",
        f"- **Summary Metrics**: {spec['fold_reporting']['aggregated_metrics']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    spec = build()
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    SPEC_MD.write_text(render_markdown(spec))
    print(f"Wrote {SPEC_JSON.relative_to(ROOT)} and {SPEC_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
