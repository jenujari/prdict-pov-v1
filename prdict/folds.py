"""Walk-forward folds, inner splits, and the holdout — the read side.

Resolves wayfinder ticket #10.
Spec at kb/fold_spec.md, derived by scripts/build_fold_spec.py.

Every boundary is read from the spec, never re-derived here, so no stage can
disagree with another about where a fold edge is.

    ./container/run.sh python -m prdict.folds    # self-check against the spec
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
class Fold:
    """One walk-forward fold.

    `inner_train` and `inner_val` partition `train`, less the inner purge.
    Early stopping and hyperparameter selection may read `inner_val`; nothing
    may read `val` until the fold's model is final.
    """

    fold: int
    train: pd.DatetimeIndex
    inner_train: pd.DatetimeIndex
    inner_val: pd.DatetimeIndex
    val: pd.DatetimeIndex


@dataclass(frozen=True)
class FoldSpec:
    """The fold geometry as generated, with the accessors stages actually use."""

    spec: dict

    @property
    def geometry(self) -> str:
        return self.spec["geometry"]

    @property
    def purge_sessions(self) -> int:
        return self.spec["purge"]["sessions"]

    @property
    def train_length(self) -> int:
        return self.spec["derivation"]["train_length_origins"]

    def folds(self, cal: TradingCalendar) -> list[Fold]:
        """The walk-forward folds, as origins on the trading calendar."""
        trainable = cal.trainable_origins()

        def block(start: str, end: str) -> pd.DatetimeIndex:
            lo, hi = pd.Timestamp(start), pd.Timestamp(end)
            return trainable[(trainable >= lo) & (trainable <= hi)]

        return [
            Fold(
                fold=f["fold"],
                train=block(f["train_start"], f["train_end"]),
                inner_train=block(f["inner_train_start"], f["inner_train_end"]),
                inner_val=block(f["inner_val_start"], f["inner_val_end"]),
                val=block(f["val_start"], f["val_end"]),
            )
            for f in self.spec["walk_forward"]["folds"]
        ]

    def holdout(self, cal: TradingCalendar) -> pd.DatetimeIndex:
        """The origins reserved for the final three-model comparison."""
        h = self.spec["holdout"]
        trainable = cal.trainable_origins()
        lo, hi = pd.Timestamp(h["start_date"]), pd.Timestamp(h["end_date"])
        return trainable[(trainable >= lo) & (trainable <= hi)]

    def final_train(self, cal: TradingCalendar) -> pd.DatetimeIndex:
        """The training block the holdout-scored models are refit on.

        Same rolling length and same purge as every fold, so the holdout
        comparison is not a different experiment from the folds.
        """
        h = self.spec["holdout"]
        trainable = cal.trainable_origins()
        lo, hi = pd.Timestamp(h["final_train_start"]), pd.Timestamp(h["final_train_end"])
        return trainable[(trainable >= lo) & (trainable <= hi)]

    def final_inner_split(self, cal: TradingCalendar) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """Inner train/val for the holdout refit — `final_train()` is a training
        block too, so the same rule that gives every fold an `inner_train`/
        `inner_val` (`kb/fold_spec.md`: "the last 15% of *each* training block")
        applies here, not only to the five rolling folds. Early stopping and
        hyperparameter selection for the holdout-scored models read this and
        nothing else, exactly as `Fold.inner_val` does.

        `holdout.final_inner_val_start/end` are already recorded in the spec;
        `inner_train` is derived the same way `scripts/build_fold_spec.py`'s own
        `inner_split()` computes it for the rolling folds — everything in
        `final_train()` up to `purge_sessions` sessions before `inner_val` starts.
        """
        h = self.spec["holdout"]
        train = self.final_train(cal)
        lo, hi = pd.Timestamp(h["final_inner_val_start"]), pd.Timestamp(h["final_inner_val_end"])
        inner_val = train[(train >= lo) & (train <= hi)]
        cutoff = cal.position(inner_val[0]) - self.purge_sessions
        inner_train = train[train.map(cal.position) <= cutoff]
        return inner_train, inner_val

    def concatenated_val_origins(self, cal: TradingCalendar) -> pd.DatetimeIndex:
        """The out-of-sample series the headline scorecard is computed on.

        The validation blocks are contiguous and disjoint, so this is one
        unbroken run of origins. See the fold-boundary rule in the spec for how
        positions are handled where one model hands over to the next.
        """
        parts = [f.val for f in self.folds(cal)]
        return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]

    def effective_sample_size(self, cal: TradingCalendar) -> int:
        """Roughly independent observations in the concatenated series.

        Consecutive origins share 9 of their 10 label days, so the origin count
        overstates the evidence by about an order of magnitude. Significance
        tests and confidence intervals use this, not `len(...)`.
        """
        return len(self.concatenated_val_origins(cal)) // cal.horizon


def load_fold_spec(path: Path = SPEC_JSON) -> FoldSpec:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/build_fold_spec.py first.")
    return FoldSpec(spec=json.loads(path.read_text()))


def main() -> None:
    cal = load_calendar()
    fs = load_fold_spec()

    print(f"Geometry: {fs.geometry}, train {fs.train_length} origins, purge {fs.purge_sessions}")
    for f in fs.folds(cal):
        print(
            f"  Fold {f.fold}: train {len(f.train)} "
            f"({f.train[0].date()} -> {f.train[-1].date()}) | "
            f"inner {len(f.inner_train)}/{len(f.inner_val)} | "
            f"val {len(f.val)} ({f.val[0].date()} -> {f.val[-1].date()})"
        )

    series = fs.concatenated_val_origins(cal)
    print(
        f"  Out-of-sample series: {len(series)} origins "
        f"({series[0].date()} -> {series[-1].date()}), "
        f"~{fs.effective_sample_size(cal)} independent"
    )
    ft, ho = fs.final_train(cal), fs.holdout(cal)
    print(f"  Final train: {len(ft)} ({ft[0].date()} -> {ft[-1].date()})")
    print(f"  Holdout:     {len(ho)} ({ho[0].date()} -> {ho[-1].date()})")

    fit, fiv = fs.final_inner_split(cal)
    print(
        f"  Final inner: {len(fit)}/{len(fiv)} "
        f"({fiv[0].date()} -> {fiv[-1].date()})"
    )
    assert fit[-1] < fiv[0], "final inner_train must end before final inner_val starts"
    assert fs.purge_sessions <= cal.position(fiv[0]) - cal.position(fit[-1])


if __name__ == "__main__":
    main()
