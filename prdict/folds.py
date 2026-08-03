"""Walk-forward cross-validation folds and holdout splits.

Resolves wayfinder ticket #10.
Spec at kb/fold_spec.md, derived by scripts/build_fold_spec.py.

    ./container/run.sh python -m prdict.folds    # self-check fold boundaries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from prdict.trading_calendar import TradingCalendar, load_calendar

ROOT = Path(__file__).resolve().parent.parent
SPEC_JSON = ROOT / "kb" / "fold_spec.json"


@dataclass(frozen=True)
class FoldSplit:
    """Origins for a single fold split."""

    fold: int
    train_origins: pd.DatetimeIndex
    val_origins: pd.DatetimeIndex
    purge_origins: pd.DatetimeIndex
    embargo_origins: pd.DatetimeIndex


@dataclass(frozen=True)
class WalkForwardSpec:
    """Spec and generator for walk-forward folds and holdout."""

    spec_dict: dict

    def get_holdout_origins(self, cal: TradingCalendar) -> pd.DatetimeIndex:
        trainable = cal.trainable_origins()
        start = pd.Timestamp(self.spec_dict["holdout"]["start_date"])
        end = pd.Timestamp(self.spec_dict["holdout"]["end_date"])
        return trainable[(trainable >= start) & (trainable <= end)]

    def get_folds(self, cal: TradingCalendar) -> list[FoldSplit]:
        trainable = cal.trainable_origins()
        splits = []

        for f in self.spec_dict["walk_forward"]["folds"]:
            val_start = pd.Timestamp(f["val_start"])
            val_end = pd.Timestamp(f["val_end"])

            val_idx_start = cal.position(val_start)
            val_idx_end = cal.position(val_end)

            purge_cutoff_idx = val_idx_start - f["purge_pre_sessions"]
            embargo_end_idx = val_idx_end + f["embargo_post_sessions"]

            val_origins = trainable[(trainable >= val_start) & (trainable <= val_end)]
            train_origins = trainable[trainable.map(cal.position) <= purge_cutoff_idx]

            purge_origins = trainable[
                (trainable.map(cal.position) > purge_cutoff_idx)
                & (trainable.map(cal.position) < val_idx_start)
            ]
            embargo_origins = trainable[
                (trainable.map(cal.position) > val_idx_end)
                & (trainable.map(cal.position) <= embargo_end_idx)
            ]

            splits.append(
                FoldSplit(
                    fold=f["fold"],
                    train_origins=train_origins,
                    val_origins=val_origins,
                    purge_origins=purge_origins,
                    embargo_origins=embargo_origins,
                )
            )

        return splits


def load_fold_spec(path: Path = SPEC_JSON) -> WalkForwardSpec:
    if not path.exists():
        raise FileNotFoundError(f"Fold spec {path} does not exist. Run scripts/build_fold_spec.py first.")
    return WalkForwardSpec(spec_dict=json.loads(path.read_text()))


def main() -> None:
    cal = load_calendar()
    wf = load_fold_spec()

    print("Walk-Forward Fold Spec Verification:")
    holdout = wf.get_holdout_origins(cal)
    print(f"  Holdout origins: {len(holdout)} ({holdout[0].date()} -> {holdout[-1].date()})")

    folds = wf.get_folds(cal)
    for fold in folds:
        print(
            f"  Fold {fold.fold}: Train {len(fold.train_origins)} ({fold.train_origins[0].date()} -> {fold.train_origins[-1].date()}) | "
            f"Val {len(fold.val_origins)} ({fold.val_origins[0].date()} -> {fold.val_origins[-1].date()})"
        )


if __name__ == "__main__":
    main()
