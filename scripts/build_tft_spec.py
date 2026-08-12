"""Fix the TFT's training contract in writing, before any model runs.

Resolves wayfinder ticket #39's config/budget half. Like `build_xgboost_spec.py`,
this derives nothing from data — it is a **design decision**, written down and
committed *before* `scripts/train_tft.py` ever runs a fit, per `kb/fold_spec.md`'s
rule: the search space and budget are fixed in writing beforehand and never
expanded after seeing inner-validation results.

TFT is far more expensive per fit than XGBoost (`kb/research/tft-stack-selection.md`:
145.7s/epoch already measured, at a smaller decode=10 shape). A 16-trial search
like #38's XGBoost search is not affordable here — the trial count shrinks, not
the principle: a real 4-trial random search, over the three hyperparameters the
research doc's own cost analysis identifies as the load-bearing ones
(`hidden_size`, `attention_head_size`, `dropout`), searched only on the inner
fit. The winning config alone gets the expensive full-block refit — the same
search-then-refit shape #38 uses for XGBoost, so both models are held to the
same discipline.

`hidden_continuous_size`, `lstm_layers`, `share_single_variable_networks` and
`batch_size` are fixed, not searched: the research doc is explicit that
`hidden_continuous_size` is the single biggest FLOP lever for reals and should
not be raised before `hidden_size`, and `share_single_variable_networks=True`
roughly halves VSN parameters regardless of the other three (matters on
~2470 training origins, unrelated to what's being searched).

    uv run python scripts/build_tft_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "kb" / "tft_spec.json"
OUT_MD = ROOT / "kb" / "tft_spec.md"

# Fixed here, once. `scripts/train_tft.py` reads this file rather than defining
# its own copy, so the space training actually uses is provably the one
# committed before any fit ran.
SEARCH_SPACE = {
    "hidden_size": [8, 16, 32],
    "attention_head_size": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
}
N_TRIALS = 4
SEARCH_SEED = 0

FIXED_ARCHITECTURE = {
    "hidden_continuous_size": 8,
    "lstm_layers": 1,
    "share_single_variable_networks": True,
}
BATCH_SIZE = 64
LOSS = "MAE"
TARGET_NORMALIZER = "identity"
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
NUM_WORKERS = 0
NUM_THREADS = 6


def build() -> dict:
    return {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/39",
        "core": {
            "max_encoder_length": 60,
            "max_prediction_length": 30,
            "scored_steps": 10,
            "loss": LOSS,
            "loss_note": (
                "point prediction, not QuantileLoss — matches #38's XGBoost "
                "predictions (also a point vector) and #14's evaluation input "
                "shape exactly, no quantile-to-point reduction to justify"
            ),
            "target_normalizer": TARGET_NORMALIZER,
            "target_normalizer_note": (
                "decoder carries raw log returns, matching XGBoost's "
                "untransformed target — the pattern dataset.py's own "
                "self-check already uses"
            ),
            "learning_rate": LEARNING_RATE,
        },
        "fixed_architecture": FIXED_ARCHITECTURE,
        "batch_size": BATCH_SIZE,
        "search": {
            "space": SEARCH_SPACE,
            "n_trials": N_TRIALS,
            "seed": SEARCH_SEED,
            "budget_rule": (
                "fixed in writing here before any model runs; not expanded "
                "after seeing inner validation results (kb/fold_spec.md's "
                "rule, applied to the TFT's own search exactly as #38 "
                "applies it to XGBoost's) — shrunk to 4 trials from #38's "
                "16 because a TFT fit costs orders of magnitude more per "
                "trial, not because the search is any less real"
            ),
            "reads": "inner_train/inner_val only (Fold.inner_train/inner_val, or FoldSpec.final_inner_split for the holdout refit) — never the outer val or the holdout",
            "not_searched": (
                "hidden_continuous_size, lstm_layers, share_single_variable_networks, "
                "batch_size — fixed_architecture above; the research doc identifies "
                "hidden_size as the primary FLOP/capacity lever and hidden_continuous_size "
                "as secondary, so only the former is searched"
            ),
        },
        "early_stopping": {
            "monitor": "inner-val loss",
            "patience": PATIENCE,
            "max_epochs": MAX_EPOCHS,
            "max_epochs_note": "an upper bound, not a target — expected to stop well before this given ~2470 training origins",
        },
        "selection_and_refit": {
            "rule": "best trial by inner-val loss",
            "refit": (
                "the winning config refits fresh (new weights, same "
                "hyperparameters) on the ENTIRE block (fold.train, or "
                "final_train for the holdout arm) for exactly best_epoch "
                "epochs — no early stopping, so no further validation data "
                "is read. Same two-phase search-then-refit shape as #38's "
                "XGBoost contract."
            ),
        },
        "cpu": {
            "num_workers": NUM_WORKERS,
            "num_workers_note": "worker IPC often costs more than TimeSeriesDataSet.__getitem__ here (research doc)",
            "num_threads": NUM_THREADS,
            "accelerator": "cpu",
        },
        "runs": {
            "contexts": "5 folds x 2 sets (predict on fold.val) + 1 final refit x 2 sets (predict on holdout) = 12 prediction sets",
            "total_fits": f"12 x ({N_TRIALS} search trials + 1 final refit) = {12 * (N_TRIALS + 1)}",
        },
    }


def render_markdown(spec: dict) -> str:
    s = spec["search"]
    c = spec["core"]
    lines: list[str] = []
    a = lines.append

    a("# TFT training contract")
    a("")
    a("Generated by `scripts/build_tft_spec.py`. Do not hand-edit — rerun the script.")
    a("")
    a(f"Resolves [#39]({spec['resolves']})'s config/budget half. A **design decision**, "
      "not a data-derived spec — written and committed before `scripts/train_tft.py` "
      "runs a single fit, per `kb/fold_spec.md`'s rule that the search space and budget "
      "are fixed in writing beforehand and never expanded after seeing results.")
    a("")
    a("## Core (encoder/decoder already fixed by #11 / docs/adr/0002)")
    a("")
    a(f"- `max_encoder_length={c['max_encoder_length']}`, `max_prediction_length={c['max_prediction_length']}`, "
      f"scored on the first {c['scored_steps']} decoder steps.")
    a(f"- Loss: `{c['loss']}`. {c['loss_note']}.")
    a(f"- Target normalizer: `{c['target_normalizer']}`. {c['target_normalizer_note']}.")
    a(f"- Learning rate: `{c['learning_rate']}` (Adam, library default via `from_dataset`).")
    a("")
    a("## Fixed architecture (not searched)")
    a("")
    fa = spec["fixed_architecture"]
    for k, v in fa.items():
        a(f"- `{k}={v}`")
    a(f"- `batch_size={spec['batch_size']}`")
    a("")
    a("## Hyperparameter search")
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
    a(f"Not searched: {s['not_searched']}.")
    a("")
    a("## Early stopping")
    a("")
    es = spec["early_stopping"]
    a(f"Monitor: {es['monitor']}. `patience={es['patience']}`, `max_epochs={es['max_epochs']}` "
      f"({es['max_epochs_note']}).")
    a("")
    a("## Selection and refit")
    a("")
    sr = spec["selection_and_refit"]
    a(f"Selection: {sr['rule']}. Refit: {sr['refit']}")
    a("")
    a("## CPU")
    a("")
    cpu = spec["cpu"]
    a(f"`num_workers={cpu['num_workers']}` ({cpu['num_workers_note']}), "
      f"`num_threads={cpu['num_threads']}`, `accelerator=\"{cpu['accelerator']}\"`.")
    a("")
    a("## Scope")
    a("")
    r = spec["runs"]
    a(f"{r['contexts']}. Total TFT fits: {r['total_fits']}.")
    a("")
    return "\n".join(lines)


def main() -> None:
    spec = build()
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(spec))
    print(f"loss={spec['core']['loss']}  "
          f"search: {spec['search']['n_trials']} trials x {len(spec['search']['space'])} params  "
          f"total fits: {spec['runs']['total_fits']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
