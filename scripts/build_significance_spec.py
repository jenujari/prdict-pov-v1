"""Fix the significance test's contract in writing, before any permutation runs.

Resolves wayfinder ticket #54's design half. Like `build_xgboost_spec.py`/
`build_tft_spec.py`, this derives nothing from data — it is a **design
decision**, written down and committed *before* `scripts/run_significance_test.py`
ever runs a permutation, per `kb/fold_spec.md`'s rule extended to this ticket:
the null-construction method and the permutation budget are fixed in writing
beforehand and not adjusted after seeing how the null distribution comes out.

What's under test: #40's one flagged-but-unresolved result — XGBoost set2's
holdout Sharpe (0.426) beating buy-and-hold (0.288), while that same arm's own
aggregate-across-folds Sharpe (0.360) sits well under baseline's (0.643).

Null construction is **target permutation ("y-scrambling")**: refit the exact
real pipeline (`kb/xgboost_run.md`'s `final`/`set2` winning hyperparameters
and fixed round count — no search, that choice is already made) on
label-shuffled `final_train` data, predict on the real holdout features,
score against the real holdout returns through `prdict/evaluation.py`
unchanged. Each origin's own 10-step target vector is shuffled as a unit
(reassigned to a different origin's feature row) rather than shuffled
step-by-step, so the null preserves the target's own autocorrelation/
distribution and only destroys the feature-to-target mapping — that mapping
is exactly what's under test.

    uv run python scripts/build_significance_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "kb" / "significance_spec.json"
OUT_MD = ROOT / "kb" / "significance_spec.md"

# Fixed here, once. scripts/run_significance_test.py reads this file rather
# than defining its own copy, so the test actually run is provably the one
# committed before any permutation ran.
TARGET = {
    "model": "xgboost",
    "set_name": "set2",
    "context": "final",
    "metric": "sharpe",
    "observed_value": 0.426,
    "source": "kb/final_comparison.md's holdout table / kb/xgboost_run.md's final/set2 row",
}
WINNING_PARAMS = {
    "max_depth": 3,
    "eta": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "min_child_weight": 5,
    "reg_lambda": 5.0,
}
BEST_ITERATION = 24  # kb/xgboost_run.md final/set2 -- refit round count is best_iteration + 1
N_PERMUTATIONS = 200
SEED_BASE = 0  # permutation i uses seed SEED_BASE + i
ALPHA = 0.05  # conventional significance threshold, stated up front


def build() -> dict:
    return {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/54",
        "target": TARGET,
        "null_construction": {
            "method": "target_permutation",
            "method_note": (
                "a.k.a. y-scrambling / White's Reality Check style null for a single "
                "strategy: refit the exact real pipeline on label-shuffled training "
                "data, predict on real (unshuffled) holdout features, score against "
                "real (unshuffled) holdout returns through the unchanged #14 protocol. "
                "If this routinely reaches the observed Sharpe even with noise labels, "
                "the observed Sharpe isn't evidence of anything."
            ),
            "shuffle_unit": (
                "each final_train origin's own 10-step target vector is permuted as a "
                "whole (reassigned to a different origin's feature row) -- never "
                "shuffled step-by-step within a vector -- so the null preserves the "
                "target's own autocorrelation/distribution and only destroys the "
                "feature-to-target mapping, which is exactly what's under test"
            ),
            "features": "real, fold-scoped set2 PCA matrix for final_train -- built once, reused across all permutations (the PCA fit depends only on features, never on labels)",
            "model_config": "real winning hyperparameters and fixed round count from kb/xgboost_run.md's final/set2 row -- no search per permutation, that choice is already made and is not re-litigated here",
            "prediction_and_scoring": "real (unshuffled) holdout features and real (unshuffled) holdout returns, scored through prdict/evaluation.py unchanged -- only the training labels are ever shuffled",
        },
        "winning_params": WINNING_PARAMS,
        "best_iteration": BEST_ITERATION,
        "num_boost_round": BEST_ITERATION + 1,
        "budget": {
            "n_permutations": N_PERMUTATIONS,
            "seed_base": SEED_BASE,
            "seed_rule": "permutation i uses seed SEED_BASE + i, i = 0..N_PERMUTATIONS-1",
            "budget_rule": (
                "fixed in writing here before any permutation runs; not extended "
                "after seeing how the null distribution comes out (kb/fold_spec.md's "
                "fixed-before-you-look rule, applied to this ticket's own budget)"
            ),
        },
        "p_value": {
            "rule": "one-sided: fraction of null Sharpes >= the observed value",
            "alpha": ALPHA,
        },
        "secondary": {
            "test": "binomial test of each model/set arm's directional hit-rate against 0.5",
            "note": "no retraining needed -- reuses kb/final_comparison.md's already-computed hit-rates and origin counts directly",
        },
    }


def render_markdown(spec: dict) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Significance test contract")
    a("")
    a("Generated by `scripts/build_significance_spec.py`. Do not hand-edit — rerun the script.")
    a("")
    a(f"Resolves [#54]({spec['resolves']})'s design half. A **design decision**, not a "
      "data-derived spec — written and committed before `scripts/run_significance_test.py` "
      "runs a single permutation, per `kb/fold_spec.md`'s rule that the method and budget "
      "are fixed in writing beforehand and never adjusted after seeing results.")
    a("")
    a("## Target")
    a("")
    t = spec["target"]
    a(f"`{t['model']}`/`{t['set_name']}`/`{t['context']}` — observed {t['metric']} = "
      f"**{t['observed_value']}**. Source: {t['source']}.")
    a("")
    a("## Null construction")
    a("")
    nc = spec["null_construction"]
    a(f"**Method: {nc['method']}.** {nc['method_note']}")
    a("")
    a(f"- Shuffle unit: {nc['shuffle_unit']}.")
    a(f"- Features: {nc['features']}.")
    a(f"- Model config: {nc['model_config']}.")
    a(f"- Prediction/scoring: {nc['prediction_and_scoring']}.")
    a("")
    a("## Fixed model config (from the real run, not re-searched)")
    a("")
    a("| Parameter | Value |")
    a("|-----------|-------|")
    for k, v in spec["winning_params"].items():
        a(f"| `{k}` | {v} |")
    a(f"| `num_boost_round` | {spec['num_boost_round']} (best_iteration {spec['best_iteration']} + 1) |")
    a("")
    a("## Budget")
    a("")
    b = spec["budget"]
    a(f"**N = {b['n_permutations']} permutations**, seeds `{b['seed_rule']}`. {b['budget_rule']}.")
    a("")
    a("## p-value")
    a("")
    p = spec["p_value"]
    a(f"{p['rule']}. Conventional threshold stated up front: alpha = {p['alpha']}.")
    a("")
    a("## Secondary test")
    a("")
    s = spec["secondary"]
    a(f"{s['test']}. {s['note']}.")
    a("")
    return "\n".join(lines)


def main() -> None:
    spec = build()
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(spec))
    print(f"target: {spec['target']['model']}/{spec['target']['set_name']}/{spec['target']['context']} "
          f"observed_sharpe={spec['target']['observed_value']}")
    print(f"null: {spec['null_construction']['method']}  N={spec['budget']['n_permutations']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
