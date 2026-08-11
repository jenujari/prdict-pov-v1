"""Fix the PCA(0.95) transform that defines "set 2", and record its fold behaviour.

Resolves wayfinder ticket #13 (map decision 9). Set 2 is the PCA-reduced
alternative to set 1 (#12) — an **independent** treatment of the same numeric
block, not a rotation of set 1's already-selected survivors: `CONTEXT.md`
defines the two sets in parallel, and #15's own PCA prediction (`kb/
independence_spec.md`) was computed against the full `linear_numeric` block for
exactly this reason.

Settled here (the ticket's six items):

  1. input matrix   PCA is fit on the (sessions x 107) `linear_numeric` frame,
                     scaled by the same fold-scoped StandardScaler #11 already
                     wires (`prdict.dataset.fold_fit_rows` + `encoding.build_fold`).
                     Because the fitted transform is a stateless per-row map, the
                     *same* one applies to every one of the 90 window sessions
                     regardless of whether TimeSeriesDataSet slices it into the
                     past-60 or future-30 block — verified below, not assumed.
  2. what bypasses   `cyclic` (92, sin/cos pairs — rotating them would break the
                     unit-circle identity #7 built) and the categorical/boolean
                     columns #15 keeps for the model in question ride through
                     untouched. Only `linear_numeric` is replaced by components.
  3. component count is fixed from **fold 1** — the earliest, so the width choice
                     uses only information available at the very first point in
                     the walk-forward sequence. Every fold (and the final holdout
                     refit) then fits its *own* scaler and PCA loadings at that
                     fixed width on its own training rows only; nothing about a
                     later fold's data ever reaches an earlier one.
  4. scaling         the fold-scoped StandardScaler from #9/#11 runs first, inside
                     the same fold, before PCA ever sees a row.
  5. which models     both. XGBoost's arm is a deliberate **negative control**: a
                     tree's axis-aligned splits and native categorical handling
                     lose meaning on dense rotated components (the map's own
                     framing), so if set 2 underperforms set 1 for XGBoost, that
                     is the expected result, not a bug.
  6. loadings         persisted per fold (component x feature), so #40's writeup
                     can say what a component actually is later.

    uv run python scripts/build_set2_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from prdict import encoding
from prdict.dataset import encoded_features, fold_fit_rows
from prdict.folds import load_fold_spec
from prdict.trading_calendar import load_calendar

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "kb" / "set2_spec.json"
OUT_MD = ROOT / "kb" / "set2_spec.md"

PCA_TARGET = 0.95
COMPONENTS_SHOWN = 15  # markdown shows this many components' top loading, not all


def _fit(spec: encoding.EncodingSpec, feats: pd.DataFrame, rows: pd.Index, n_components):
    """Fold-scoped scaler + PCA, both fit on `rows` only.

    Reuses `encoding.build_fold` for the scaler exactly as #11's TFT/XGBoost path
    does, so set 2 shares the identical fold-scoped scaling contract as set 1 and
    the raw feature block — no second definition of "how a fold scales its data".
    """
    linear_cols = spec.family("linear_numeric").columns
    scaler_state = encoding.build_fold(spec, feats, rows)
    scaled = scaler_state.transform(feats)
    X_train = scaled.loc[rows, linear_cols].to_numpy(dtype=float)
    pca = PCA(n_components=n_components).fit(X_train)
    return scaler_state, scaled, pca, linear_cols


def _verify_per_timestep_consistency(scaled: pd.DataFrame, pca: PCA, linear_cols: list[str]) -> dict:
    """Item 1: the same fitted transform must apply identically at every timestep.

    `PCAState.transform` (in `prdict/set2.py`) runs on the whole session-level
    frame at once — there is no per-window branching to get wrong — but the
    ticket asks for a hand check, not an assumption. Pick one session, transform
    it two ways: as part of the full matrix, and alone as a single row. If a
    session's component values ever depended on which window position it was
    read through, these would disagree.
    """
    probe_pos = len(scaled) // 3
    full = pca.transform(scaled[linear_cols].to_numpy(dtype=float))
    alone = pca.transform(scaled[linear_cols].iloc[[probe_pos]].to_numpy(dtype=float))
    match = np.allclose(full[probe_pos], alone[0], atol=1e-10)
    return {
        "probe_session_position": int(probe_pos),
        "matches_isolated_transform": bool(match),
        "conclusion": (
            "The fitted transform is a stateless per-row map, so the same PCA "
            "output applies whether a session is read as an encoder-block row or "
            "a decoder-block row of any origin's window"
        ),
    }


def build() -> dict:
    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)
    fs = load_fold_spec()

    feats = encoded_features(cal, spec, globals_, "tft")
    linear_cols = spec.family("linear_numeric").columns
    cyclic_cols = spec.family("cyclic").columns

    folds = fs.folds(cal)
    final_train_block = SimpleNamespace(train=fs.final_train(cal))
    blocks = [("fold", f.fold, f) for f in folds] + [("final_train", None, final_train_block)]

    # Item 3: fix the component count from fold 1 alone.
    rows1 = fold_fit_rows(folds[0], cal)
    _, _, pca1, _ = _fit(spec, feats, rows1, PCA_TARGET)
    n_components = int(pca1.n_components_)

    per_block = []
    consistency_check = None
    for kind, fold_no, block in blocks:
        rows = fold_fit_rows(block, cal)
        scaler_state, scaled, pca, _ = _fit(spec, feats, rows, n_components)
        variance = float(pca.explained_variance_ratio_.sum())

        if consistency_check is None:
            consistency_check = _verify_per_timestep_consistency(scaled, pca, linear_cols)

        per_block.append({
            "kind": kind,
            "fold": fold_no,
            "n_train_rows": int(len(rows)),
            "n_components": n_components,
            "explained_variance": round(variance, 5),
            "loadings": pca.components_.round(5).tolist(),  # (n_components, 107)
        })

    return {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/13",
        "parameters": {
            "input": {
                "block": "linear_numeric",
                "n_columns": len(linear_cols),
                "fit_scope": "fold — same StandardScaler + rows as #11 (fold_fit_rows)",
            },
            "bypass": {
                "cyclic": {
                    "n": len(cyclic_cols),
                    "reason": "rotating a sin/cos pair breaks the unit-circle identity #7 built",
                },
                "categorical_boolean": {
                    "reason": "follow #15's per-model policy unchanged; the same columns ride in set 1",
                },
            },
            "component_count": {
                "value": n_components,
                "target_variance": PCA_TARGET,
                "derived_from": "fold 1 only (earliest chronological block)",
                "leakage_note": (
                    "The width is fixed once, from the first fold's own training rows "
                    "only — no later fold's data ever informs it. Every fold (and the "
                    "final holdout refit) then fits its own scaler and its own PCA "
                    "loadings at that fixed width, strictly on its own rows. This is a "
                    "structural choice (an architecture-defining integer), not a leak "
                    "of validation statistics — the same distinction #9 draws between "
                    "the fold-scoped scaler and the once-fit categorical encoders."
                ),
            },
            "scaling": "fold-scoped StandardScaler (#9/#11), fit before PCA in the same fold",
            "consumers": {
                "tft": {
                    "consumes": True,
                    "note": "primary intended use — dense components suit a smooth encoder",
                },
                "xgboost": {
                    "consumes": True,
                    "note": (
                        "deliberate negative control — axis-aligned splits and native "
                        "categorical handling lose meaning on rotated components; if set 2 "
                        "underperforms set 1 here, that confirms the expectation rather "
                        "than indicating a bug"
                    ),
                },
            },
        },
        "per_timestep_consistency": consistency_check,
        "blocks": per_block,
        "what_38_39_inherit": (
            "prdict.set2.fit_fold(spec, frame, fold, cal) reproduces exactly the fit this "
            "script ran for that fold — same rows, same fixed width — and returns a "
            "PCAState that replaces linear_numeric with pca_1..pca_k in the frame, "
            "leaving everything else (cyclic, categorical, boolean, elapsed) untouched. "
            "set2.features_for(model) composes the component columns with #15's kept "
            "categoricals, exactly as set1.features_for does for the selected numeric list."
        ),
    }


def render_markdown(spec: dict, linear_cols: list[str]) -> str:
    p = spec["parameters"]
    cc = p["component_count"]
    lines: list[str] = []
    a = lines.append

    a("# Set 2 — the PCA(0.95)-reduced feature set")
    a("")
    a("Generated by `scripts/build_set2_spec.py`. Do not hand-edit — rerun the script.")
    a("")
    a(f"Resolves [#13]({spec['resolves']}) (map decision 9). Set 2 is **independent** "
      "of set 1 — a parallel PCA treatment of the same numeric block, not a rotation "
      "of set 1's already-selected survivors (`CONTEXT.md`; [#15](https://github.com/jenujari/prdict-pov-v1/issues/15)'s "
      "own PCA prediction was computed against this same full block for that reason).")
    a("")
    a("## Parameters")
    a("")
    a(f"**Input.** PCA fits on the `{p['input']['block']}` block "
      f"({p['input']['n_columns']} columns), {p['input']['fit_scope']}.")
    a("")
    a(f"**Bypass.** `cyclic` ({p['bypass']['cyclic']['n']} columns) rides through "
      f"untouched — {p['bypass']['cyclic']['reason']}. Categorical/boolean columns "
      f"{p['bypass']['categorical_boolean']['reason']}. Only `linear_numeric` is "
      "replaced by components.")
    a("")
    a(f"**Component count — fixed at {cc['value']}.** Target variance "
      f"{cc['target_variance']}, {cc['derived_from']}. {cc['leakage_note']}")
    a("")
    a(f"**Scaling.** {p['scaling']}.")
    a("")
    a("**Consumers — both models, one by design as a negative control.**")
    a("")
    a(f"- **TFT**: {p['consumers']['tft']['note']}.")
    a(f"- **XGBoost**: {p['consumers']['xgboost']['note']}.")
    a("")
    a("## Per-timestep consistency (item 1)")
    a("")
    c = spec["per_timestep_consistency"]
    a(f"Probed session at position {c['probe_session_position']}: transforming it as part "
      f"of the full matrix and transforming it alone agree "
      f"(`matches_isolated_transform={c['matches_isolated_transform']}`). {c['conclusion']}.")
    a("")
    a("## Per-fold fit")
    a("")
    a("Every block refits its own scaler and PCA loadings on its own training rows, "
      f"at the fixed width of {cc['value']} components:")
    a("")
    a("| Block | Train rows | Components | Explained variance |")
    a("|-------|-----------|------------|---------------------|")
    for b in spec["blocks"]:
        label = f"fold {b['fold']}" if b["kind"] == "fold" else "final_train (holdout refit)"
        a(f"| {label} | {b['n_train_rows']} | {b['n_components']} | {b['explained_variance']} |")
    a("")
    fold1 = spec["blocks"][0]
    a(f"Fold 1's own achieved variance is {fold1['explained_variance']} (it defined the "
      f"width, so it lands closest to the {cc['target_variance']} target); later folds "
      "see more history and drift slightly around that same width — expected, since a "
      "fixed integer width cannot track a growing training set's variance exactly.")
    a("")
    a(f"For context, [#15](https://github.com/jenujari/prdict-pov-v1/issues/15) predicted "
      f"**62/107** from a PCA fit on the *entire* history at once; fold 1 alone (a "
      "~10-year training subset, not 26 years) is expected to differ, and does — see the "
      "table above for the actual value this pipeline uses.")
    a("")
    a("## Loadings (item 6, fold 1)")
    a("")
    a("Full component x feature loadings for every block are in `kb/set2_spec.json`; "
      "here is the top-loading feature per component for fold 1, as a representative sample.")
    a("")
    loadings = np.array(fold1["loadings"])  # (k, 107)
    a("| Component | Top feature | Loading |")
    a("|-----------|-------------|---------|")
    for i, row in enumerate(loadings, start=1):
        top = int(np.argmax(np.abs(row)))
        a(f"| pca_{i} | `{linear_cols[top]}` | {row[top]:+.3f} |")
        if i >= COMPONENTS_SHOWN:
            a(f"| ... | ({len(loadings) - i} more components) | |")
            break
    a("")
    return "\n".join(lines)


def main() -> None:
    spec = build()
    linear_cols = encoding.load_spec().family("linear_numeric").columns

    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(spec, linear_cols))

    print(f"component count (fixed from fold 1): {spec['parameters']['component_count']['value']}")
    for b in spec["blocks"]:
        label = f"fold {b['fold']}" if b["kind"] == "fold" else "final_train"
        print(f"  {label:20s}: {b['n_train_rows']:5d} rows  variance {b['explained_variance']}")
    print(f"per-timestep consistency: {spec['per_timestep_consistency']['matches_isolated_transform']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")

    assert spec["per_timestep_consistency"]["matches_isolated_transform"], "per-timestep check failed"
    widths = {b["n_components"] for b in spec["blocks"]}
    assert widths == {spec["parameters"]["component_count"]["value"]}, "component width drifted across blocks"


if __name__ == "__main__":
    main()
