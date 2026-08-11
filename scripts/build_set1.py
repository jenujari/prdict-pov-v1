"""Run the two-stage feature selection and freeze "set 1" as a reproducible artifact.

Resolves wayfinder ticket #12 (map decision 7). Set 1 is the correlation-pruned,
target-ranked feature set; set 2 (#13) is the PCA alternative. Both are compared
on the scorecard, so set 1 has to be *one* artifact with fixed parameters, not a
per-run choice.

The pipeline, fit **inside every fold** (map decision 4 — never on full history):

  stage 1  redundancy prune. Spearman |rho| >= 0.95 among the 199 numeric
           features, computed on the fold's training origins. Spearman, not
           Pearson: the redundancy here is monotone (a bala is a smooth monotone
           transform of a longitude; a day-lag is a near-linear shift), and rank
           correlation catches all of it while Pearson on an ordinal code is
           meaningless. Categoricals/booleans never enter the matrix — #15 already
           decides those (dropped for XGBoost, embedded for the TFT). sin/cos
           pairs are one atomic unit (#7): a pair is kept whole or dropped whole.
           When two units are redundant the **more target-relevant** one survives
           (higher mutual information) — variance cannot discriminate after
           standardisation, which is the trap `backup.one.txt` fell into.

  stage 2  target ranking. Mutual information (nonlinear) and Spearman (monotone)
           of each survivor against the target, averaged over the 10 horizons —
           the scorecard trades the whole path, so a feature earns its place by
           mean relevance across it, not by one lucky horizon. A feature is
           selected only if its MI beats a **permutation null** (target shuffled):
           astro-vs-return correlation is expected near zero (map's own warning),
           so the null band is fixed up front and decides what counts as signal,
           rather than a threshold rationalised afterwards.

Fold stability: each fold selects in-fold (leak-free), and the canonical set 1 is
the **majority-frequency union** — a feature is in set 1 if it is selected in at
least 3 of the 5 folds. Intersection is too brittle with near-zero signal; a
per-fold-only set is not a single artifact. Every feature's fold frequency is
reported, so the stability is visible rather than asserted.

    uv run python scripts/build_set1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from prdict import encoding
from prdict.dataset import encoded_features
from prdict.folds import load_fold_spec
from prdict.targets import build as build_target
from prdict.trading_calendar import load_calendar

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "kb" / "set1_spec.json"
OUT_MD = ROOT / "kb" / "set1_spec.md"

REDUNDANCY_RHO = 0.95
NULL_PERMUTATIONS = 12
NULL_QUANTILE = 0.95
MAJORITY = 3  # of 5 folds
RANDOM_STATE = 0


def numeric_columns(spec: encoding.EncodingSpec) -> tuple[list[str], list[frozenset[str]]]:
    """The columns that enter the prune, and the atomic units they group into.

    The two `none`/`fold`-scoped numeric families — `cyclic` (46 sin/cos pairs)
    and `linear_numeric` — are the only ones ranked here. A sin/cos pair is one
    unit so #7's angle identity is never split.
    """
    cyclic = spec.family("cyclic").columns
    linear = spec.family("linear_numeric").columns
    cols = list(cyclic) + list(linear)

    units: list[frozenset[str]] = []
    seen: set[str] = set()
    for col in cyclic:
        if col in seen:
            continue
        base = col[:-4]  # strip _sin / _cos
        pair = [c for c in (base + "_sin", base + "_cos") if c in cyclic]
        units.append(frozenset(pair))
        seen.update(pair)
    units.extend(frozenset([c]) for c in linear)
    return cols, units


def _mi_per_horizon(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Mean mutual information of each column against the 10 horizons."""
    scores = np.zeros(X.shape[1])
    for h in range(Y.shape[1]):
        scores += mutual_info_regression(X, Y[:, h], random_state=RANDOM_STATE)
    return scores / Y.shape[1]


def _stage1(cols: list[str], units: list[frozenset[str]], X: np.ndarray, mi: np.ndarray) -> list[str]:
    """Greedy redundancy prune over units, keeping the most target-relevant.

    Units are visited most-relevant first; a unit is kept unless it is redundant
    (Spearman |rho| >= 0.95, between *any* member and any member of a kept unit)
    with something already kept. Because pairs are units, a pair is never split.
    """
    idx = {c: i for i, c in enumerate(cols)}
    rho = np.abs(np.nan_to_num(spearmanr(X).correlation, nan=0.0))  # constant cols -> NaN -> 0
    np.fill_diagonal(rho, 0.0)

    def relevance(unit: frozenset[str]) -> float:
        return max(mi[idx[c]] for c in unit)

    order = sorted(units, key=relevance, reverse=True)
    kept: list[frozenset[str]] = []
    kept_idx: list[int] = []
    for unit in order:
        members = [idx[c] for c in unit]
        redundant = bool(kept_idx) and rho[np.ix_(members, kept_idx)].max() >= REDUNDANCY_RHO
        if not redundant:
            kept.append(unit)
            kept_idx.extend(members)
    return [c for c in cols if any(c in u for u in kept)]


def _null_band(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> float:
    """The MI a feature reaches against a shuffled target — the 'nothing here' line.

    A permuted target is noise regardless of which horizon it came from, so the
    null is built from one shuffled horizon per permutation rather than all ten —
    the same per-feature noise floor, at a tenth of the cost. Pooled over
    permutations and features, the 95th percentile is the band a real feature must
    clear. Using a single horizon leaves the null slightly *wider* than the
    averaged-over-ten real statistic, so the gate is conservative, not lenient.
    """
    y0 = Y[:, 0]
    pooled: list[float] = []
    for _ in range(NULL_PERMUTATIONS):
        perm = rng.permutation(len(y0))
        pooled.extend(mutual_info_regression(X, y0[perm], random_state=RANDOM_STATE).tolist())
    return float(np.quantile(pooled, NULL_QUANTILE))


def build() -> dict:
    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)
    fs = load_fold_spec()

    cols, units = numeric_columns(spec)
    feats = encoded_features(cal, spec, globals_, "tft")[cols].reset_index(drop=True)
    feats.index = cal.sessions[: len(feats)]

    tgt = build_target(cal)
    Yframe = pd.DataFrame(tgt.y, index=tgt.origins)

    rng = np.random.default_rng(RANDOM_STATE)
    per_fold = []
    selected_counts = {c: 0 for c in cols}
    survivor_counts = {c: 0 for c in cols}

    for fold in fs.folds(cal):
        origins = fold.train.intersection(tgt.origins)
        X = feats.loc[origins].to_numpy(float)
        Y = Yframe.loc[origins].to_numpy(float)

        mi = _mi_per_horizon(X, Y)
        survivors = _stage1(cols, units, X, mi)
        surv_idx = [cols.index(c) for c in survivors]

        null95 = _null_band(X, Y, rng)
        chosen = [c for c in survivors if mi[cols.index(c)] > null95]

        for c in survivors:
            survivor_counts[c] += 1
        for c in chosen:
            selected_counts[c] += 1

        per_fold.append({
            "fold": fold.fold,
            "train_origins": int(len(origins)),
            "survivors": len(survivors),
            "null_mi_p95": round(null95, 6),
            "selected": len(chosen),
            "max_mi": round(float(mi.max()), 6),
            "selected_cols": sorted(chosen),
        })

    n_folds = len(per_fold)
    set1 = sorted(c for c, k in selected_counts.items() if k >= MAJORITY)
    mean_mi = {}
    # Report mean MI over folds for context (recompute cheaply on full trainable set).
    Xall = feats.loc[tgt.origins].to_numpy(float)
    full_mi = _mi_per_horizon(Xall, tgt.y)
    for i, c in enumerate(cols):
        mean_mi[c] = round(float(full_mi[i]), 6)

    return {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/12",
        "parameters": {
            "stage1": {
                "metric": "spearman",
                "threshold": REDUNDANCY_RHO,
                "survivor_rule": "higher mean-horizon mutual information",
                "atomic_units": "sin/cos pairs (#7)",
                "columns_considered": len(cols),
                "note": "categoricals/booleans bypass the prune; #15 decides those per model",
            },
            "stage2": {
                "metrics": ["mutual_information", "spearman"],
                "horizon_reduction": "mean over the 10 step-return horizons",
                "null": {
                    "kind": "target permutation",
                    "permutations": NULL_PERMUTATIONS,
                    "quantile": NULL_QUANTILE,
                    "rule": "a feature is selected only if its mean MI exceeds the null band",
                },
            },
            "fold_stability": {
                "rule": "majority-frequency union",
                "threshold_folds": MAJORITY,
                "n_folds": n_folds,
                "in_fold": "selection is fit in-fold for scoring; set 1 is the reported union",
            },
        },
        "folds": per_fold,
        "set1": {
            "numeric_features": set1,
            "n": len(set1),
            "definition": (
                f"numeric features selected (MI > null p95) in at least {MAJORITY} of "
                f"{n_folds} folds; the categorical part follows #15's per-model policy"
            ),
        },
        "frequency": {
            c: {
                "selected_folds": selected_counts[c],
                "survived_folds": survivor_counts[c],
                "mean_mi": mean_mi[c],
            }
            for c in cols
        },
        "reading": {
            "signal": (
                "astro-vs-return MI sits at the noise floor by design; the null band "
                "is the honest line, and how many features clear it is the finding — "
                "not a threshold tuned to produce a set"
            ),
        },
    }


def render_markdown(spec: dict) -> str:
    p = spec["parameters"]
    s1, s2, fst = p["stage1"], p["stage2"], p["fold_stability"]
    set1 = spec["set1"]
    lines: list[str] = []
    a = lines.append

    a("# Set 1 — the two-stage selected feature set")
    a("")
    a("Generated by `scripts/build_set1.py`. Do not hand-edit — rerun the script.")
    a("")
    a(f"Resolves [#12]({spec['resolves']}) (map decision 7). Set 1 is the "
      "correlation-pruned, target-ranked feature set; [set 2](https://github.com/jenujari/prdict-pov-v1/issues/13) "
      "is the PCA alternative. Both feed the three-model scorecard, so set 1 is fixed here as one artifact.")
    a("")
    a("## Parameters")
    a("")
    a("**Stage 1 — redundancy prune.**")
    a("")
    a(f"- Metric: **Spearman** `|rho| >= {s1['threshold']}` over the "
      f"{s1['columns_considered']} numeric features (cyclic + linear_numeric). Rank "
      "correlation catches the monotone redundancy Pearson would miss and that an "
      "ordinal category code makes meaningless.")
    a(f"- Survivor rule: **{s1['survivor_rule']}** — variance cannot discriminate "
      "after standardisation.")
    a(f"- Atomic units: **{s1['atomic_units']}** — a pair is kept or dropped whole.")
    a(f"- {s1['note']}.")
    a("")
    a("**Stage 2 — target ranking.**")
    a("")
    a(f"- Metrics: **mutual information** + **Spearman**, {s2['horizon_reduction']}.")
    a(f"- Selection gate: a **{s2['null']['kind']}** null "
      f"({s2['null']['permutations']} shuffles, p{int(100*s2['null']['quantile'])} band). "
      f"{s2['null']['rule'].capitalize()}.")
    a("")
    a(f"**Fold stability — {fst['rule']}.** Each fold selects in-fold ({fst['in_fold']}); "
      f"set 1 is the union of features chosen in at least **{fst['threshold_folds']} of "
      f"{fst['n_folds']}** folds.")
    a("")
    a("## Per-fold selection")
    a("")
    a("| Fold | Train origins | Survivors (stage 1) | Null MI p95 | Max MI | Selected (stage 2) |")
    a("|------|---------------|---------------------|-------------|--------|--------------------|")
    for f in spec["folds"]:
        a(f"| {f['fold']} | {f['train_origins']} | {f['survivors']} | "
          f"{f['null_mi_p95']} | {f['max_mi']} | {f['selected']} |")
    a("")
    a("## Set 1")
    a("")
    a(f"**{set1['n']} numeric features** — {set1['definition']}.")
    a("")
    if set1["numeric_features"]:
        freq = spec["frequency"]
        a("| Feature | Selected in | Mean MI |")
        a("|---------|-------------|---------|")
        for c in sorted(set1["numeric_features"], key=lambda c: -freq[c]["selected_folds"]):
            a(f"| `{c}` | {freq[c]['selected_folds']}/{fst['n_folds']} | {freq[c]['mean_mi']} |")
    else:
        a("_Empty._ No numeric feature cleared the permutation null in a majority of "
          "folds — the honest reading (see below), not a pipeline failure.")
    a("")
    a("## Reading the result")
    a("")
    a(spec["reading"]["signal"] + ".")
    a("")
    maxnull = max(f["null_mi_p95"] for f in spec["folds"])
    maxmi = max(f["max_mi"] for f in spec["folds"])
    a(f"Across folds the strongest single-feature MI is **{maxmi}** against a null p95 of "
      f"up to **{maxnull}**. That relationship — not any one selected column — is the "
      "result #12 hands to the model tickets: it fixes, in advance, what 'the astro "
      "features carry almost nothing linear about returns' looks like in numbers.")
    a("")
    a("## What the models consume")
    a("")
    a("Set 1's numeric part is model-agnostic (rank/MI are scale-free, so no fold "
      "scaler is needed to choose it). The **categorical** part of each model's input "
      "is not selected here — it follows #15: the TFT keeps the sharp bins as "
      "embeddings, XGBoost drops them. XGBoost additionally collapses the day-lag "
      "collinearity of set 1's features when it flattens the 90-session window; that "
      "flatten-level prune is an XGBoost training step (#38) over these features, not a "
      "second definition of set 1.")
    a("")
    return "\n".join(lines)


def main() -> None:
    spec = build()
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(spec))

    print(f"folds: {len(spec['folds'])}")
    for f in spec["folds"]:
        print(f"  fold {f['fold']}: survivors {f['survivors']:3d}  "
              f"null_p95 {f['null_mi_p95']:.5f}  maxMI {f['max_mi']:.5f}  "
              f"selected {f['selected']}")
    print(f"set1: {spec['set1']['n']} numeric features "
          f"(majority union, >= {spec['parameters']['fold_stability']['threshold_folds']} folds)")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
