"""Fix XGBoost's training contract in writing, before any model runs.

Resolves wayfinder ticket #38's items 1, 2, and 5. Unlike every other
`build_*_spec.py` in this repo, this script derives nothing from data — it is a
**design decision**, written down and committed *before* `scripts/train_xgboost.py`
ever runs a fit. That order is not stylistic: `kb/fold_spec.md`'s own rule is
"the hyperparameter search space and the number of configurations tried are
fixed in writing before any model runs, and are not expanded after seeing
inner validation results" — a search that is free to keep widening itself after
seeing how it scores would fit the search to a fold's ~66 independent inner
observations, not to real signal. This script is how that fixing happens.

Settled here:

  1. **Multi-output mechanics.** `multi_strategy="one_output_per_tree"` — not
     `multi_output_tree`. This looks backwards (map decision 2 asked for the
     *cumulative* target only if XGBoost's native support turned out to favour
     it), but #6's closed investigation found the opposite: `multi_output_tree`
     ignores `max_cat_to_onehot` entirely and one-hot-splits every categorical,
     which conflicts with map decision 6's whole reason for native `category`
     dtype. `one_output_per_tree` is the production choice; the target is the
     10-step **scored** vector (never the TFT's 30-step training extension).
  2. **Categoricals** ride as integer codes with `feature_types` marking `"c"`
     columns, `enable_categorical=True` — exactly what `dataset.FlatMatrix`
     already produces (#11), never pandas `category` dtype passed to `DMatrix`
     directly.
  5. **Hyperparameter search** — a fixed random-search space and a fixed
     16-trial budget, drawn once with a fixed seed and reused identically across
     every (fold-or-final, set) context. Selection reads only the inner
     validation split (`kb/fold_spec.md`); early stopping's optimal round count
     becomes the fixed `n_estimators` of the block-level refit, which touches
     no further validation data.

    uv run python scripts/build_xgboost_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "kb" / "xgboost_spec.json"
OUT_MD = ROOT / "kb" / "xgboost_spec.md"

# Fixed here, once. `scripts/train_xgboost.py` reads this file rather than
# defining its own copy, so the space training actually uses is provably the
# one committed before any fit ran.
SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "eta": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1.0, 5.0, 10.0],
}
N_TRIALS = 16
SEARCH_SEED = 0
MAX_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
EVAL_METRIC = "rmse"


def build() -> dict:
    return {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/38",
        "core": {
            "multi_strategy": "one_output_per_tree",
            "tree_method": "hist",
            "target": "10-step scored vector (prdict.targets.build(cal).y) — never the 30-step TFT training extension",
            "rejected": {
                "multi_output_tree": (
                    "ignores max_cat_to_onehot and one-hot-splits every categorical "
                    "(#6, empirically confirmed: 100/100 splits of size 1 vs sizes "
                    "1-26 under one_output_per_tree) — conflicts with map decision 6"
                ),
            },
        },
        "categoricals": {
            "encoding": "integer codes + feature_types marking \"c\" columns",
            "dmatrix_flag": "enable_categorical=True",
            "note": "matches dataset.FlatMatrix.values/feature_types exactly; never pandas category dtype passed to DMatrix directly",
        },
        "search": {
            "space": SEARCH_SPACE,
            "n_trials": N_TRIALS,
            "seed": SEARCH_SEED,
            "budget_rule": (
                "fixed in writing here before any model runs; not expanded after "
                "seeing inner validation results (kb/fold_spec.md's rule, applied "
                "to XGBoost's own search)"
            ),
            "reads": "inner_train/inner_val only (Fold.inner_train/inner_val, or FoldSpec.final_inner_split for the holdout refit) — never the outer val or the holdout",
            "max_boost_round": MAX_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "eval_metric": EVAL_METRIC,
            "eval_metric_note": "XGBoost pools rmse across all 10 outputs natively under multi-output — no custom metric needed",
        },
        "selection_and_refit": {
            "rule": "best trial by inner-val rmse",
            "refit": (
                "the winning (params, best_iteration) refits on the ENTIRE "
                "block (fold.train, or final_train for the holdout arm) with "
                "best_iteration fixed as n_estimators — no further early "
                "stopping, so no further validation data is read"
            ),
        },
        "runs": {
            "contexts": "5 folds x 2 sets (predict on fold.val) + 1 final refit x 2 sets (predict on holdout) = 12 prediction sets",
            "total_fits": "12 x (16 search trials + 1 final refit) = 204",
        },
    }


def render_markdown(spec: dict) -> str:
    s = spec["search"]
    lines: list[str] = []
    a = lines.append

    a("# XGBoost training contract")
    a("")
    a("Generated by `scripts/build_xgboost_spec.py`. Do not hand-edit — rerun the script.")
    a("")
    a(f"Resolves items 1, 2 and 5 of [#38]({spec['resolves']}). A **design decision**, "
      "not a data-derived spec — written and committed before `scripts/train_xgboost.py` "
      "runs a single fit, per `kb/fold_spec.md`'s rule that the search space and budget "
      "are fixed in writing beforehand and never expanded after seeing results.")
    a("")
    a("## Core mechanics (item 1)")
    a("")
    c = spec["core"]
    a(f"- `multi_strategy=\"{c['multi_strategy']}\"`, `tree_method=\"{c['tree_method']}\"`.")
    a(f"- Target: {c['target']}.")
    a(f"- **Not** `multi_output_tree`: {c['rejected']['multi_output_tree']}.")
    a("")
    a("## Categoricals (item 2)")
    a("")
    cat = spec["categoricals"]
    a(f"{cat['encoding']}, `{cat['dmatrix_flag']}`. {cat['note']}.")
    a("")
    a("## Hyperparameter search (item 5)")
    a("")
    a("| Parameter | Values |")
    a("|-----------|--------|")
    for k, v in s["space"].items():
        a(f"| `{k}` | {v} |")
    a("")
    a(f"**Budget: {s['n_trials']} trials**, drawn once with seed `{s['seed']}`, reused "
      f"identically across every (fold-or-final, set) context — not redrawn per context. "
      f"{s['budget_rule']}.")
    a("")
    a(f"Reads: {s['reads']}.")
    a("")
    a(f"Early stopping: `max_boost_round={s['max_boost_round']}`, "
      f"`early_stopping_rounds={s['early_stopping_rounds']}`, "
      f"`eval_metric=\"{s['eval_metric']}\"` — {s['eval_metric_note']}.")
    a("")
    a("## Selection and refit")
    a("")
    sr = spec["selection_and_refit"]
    a(f"Selection: {sr['rule']}. Refit: {sr['refit']}.")
    a("")
    a("## Scope")
    a("")
    r = spec["runs"]
    a(f"{r['contexts']}. Total XGBoost fits: {r['total_fits']}.")
    a("")
    return "\n".join(lines)


def main() -> None:
    spec = build()
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(spec))
    print(f"multi_strategy={spec['core']['multi_strategy']}  "
          f"search: {spec['search']['n_trials']} trials x {len(spec['search']['space'])} params  "
          f"total fits: {spec['runs']['total_fits']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
