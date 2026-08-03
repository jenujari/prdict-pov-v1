"""Derive the encoding contract for both model views, and assert it.

Resolves wayfinder ticket #9.

The contract is *semantic*: what each column family becomes, which dtype each
view sees, and — the load-bearing part — the scope at which each transform is
fitted. It deliberately stops short of a physical frame layout, file format or
column order; #11 owns those.

Nothing here is hand-authored. Every count is derived from `kb/column_spec.json`
and `kb/angular_spec.json` and checked against the source data, so the build
fails if an upstream spec moves underneath it.

    ./container/run.sh python scripts/build_encoding_spec.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prdict import angles, columns

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "nft50.csv"
CATEGORIES_PATH = ROOT / "categories_list.json"
OUT_JSON = ROOT / "kb" / "encoding_spec.json"
OUT_MD = ROOT / "kb" / "encoding_spec.md"

# Fit scopes. The whole point of the spec is that these three are distinct and
# that the distinction is machine-checked rather than remembered.
#
#   none   — deterministic. No parameters, nothing to fit, no leak surface.
#   global — fitted once on *declared* metadata, never on observed values.
#            Safe to share across folds precisely because no data reaches it.
#   fold   — fitted on training-fold rows only. Never global, never persisted.
SCOPES = ("none", "global", "fold")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_families(
    cspec: columns.ColumnSpec, aspec: angles.AngularSpec
) -> list[dict]:
    """The per-family contract, with counts derived from the upstream specs."""
    cyclic = list(aspec.emitted_columns)

    # Angular sources are consumed by the sin/cos transform (#7 settled that the
    # raw angle does not survive), so the linear family is exactly what
    # column_spec types as linear_numeric — the 34 kept `_dist` columns included.
    linear = list(cspec.linear_numeric)

    return [
        {
            "family": "boolean",
            "columns": list(cspec.boolean),
            "n": len(cspec.boolean),
            "fit_scope": "none",
            "transform": "passthrough",
            "dtype": {"xgboost": "int8", "tft": "float32"},
            "tft_role": "time_varying_known_real",
            "rationale": (
                "Two-level flags. Passed as 0/1 rather than as a 2-level "
                "categorical so the TFT's variable-selection network spends no "
                "embedding parameters on a bit."
            ),
        },
        {
            "family": "categorical",
            "columns": list(cspec.categorical),
            "n": len(cspec.categorical),
            "fit_scope": "global",
            "transform": "declared_levels",
            "dtype": {
                "xgboost": "pandas.Categorical(categories=declared)",
                "tft": "str + NaNLabelEncoder(add_nan=True)",
            },
            "tft_role": "time_varying_known_categorical",
            "rationale": (
                "Levels come from the declared list, never from what a fold "
                "observed, so codes and embedding rows are identical in every "
                "fold (map decision 6). Global scope is safe here because the "
                "encoder never sees a data value — only the level list."
            ),
        },
        {
            "family": "cyclic",
            "columns": cyclic,
            "n": len(cyclic),
            "fit_scope": "none",
            "transform": "sin_cos",
            "scaled": False,
            "dtype": {"xgboost": "float32", "tft": "float32"},
            "tft_role": "time_varying_known_real",
            "rationale": (
                "Already bounded in [-1, 1] and already comparable across "
                "columns. Standardising would divide each half of a pair by a "
                "different sigma, breaking sin^2 + cos^2 = 1 and with it the "
                "pair atomicity #7 requires of #12 and #13."
            ),
        },
        {
            "family": "linear_numeric",
            "columns": linear,
            "n": len(linear),
            "fit_scope": "fold",
            "transform": "StandardScaler",
            "dtype": {"xgboost": "float32", "tft": "float32"},
            "tft_role": "time_varying_known_real",
            "rationale": (
                "The only family carrying unbounded, incomparable units, and "
                "so the only one needing a fitted scaler. Fitted on training "
                "rows alone (map decision 4). Skew transforms are dropped "
                "entirely: log1p is unsafe on the "
                f"{len(cspec.log1p_unsafe)} columns that reach -1 or below, and "
                "a signed-safe variant would buy nothing a scaler does not."
            ),
        },
    ]


def build_spec(cspec: columns.ColumnSpec, aspec: angles.AngularSpec) -> dict:
    families = build_families(cspec, aspec)
    by_name = {f["family"]: f for f in families}

    sentinels = {
        col: rule["value"] for col, rule in cspec.structural_fill.items()
    }
    declared = json.loads(CATEGORIES_PATH.read_text())

    width = sum(f["n"] for f in families)
    return {
        "resolves": 9,
        "scope_note": (
            "Semantic contract only — what each family becomes and where each "
            "transform is fitted. Physical layout (file format, column order, "
            "one frame or two, the (n, 10) target block, TimeSeriesDataSet "
            "wiring) belongs to #11."
        ),
        "source": {
            "csv": CSV_PATH.name,
            "column_spec": "kb/column_spec.json",
            "angular_spec": "kb/angular_spec.json",
            "categories_list": {
                "path": CATEGORIES_PATH.name,
                "sha256": _sha256(CATEGORIES_PATH),
                "mutated": False,
            },
        },
        "fit_scopes": {
            "none": {
                "meaning": "Deterministic; no parameters, no leak surface.",
                "families": [f["family"] for f in families if f["fit_scope"] == "none"],
                "persisted": False,
            },
            "global": {
                "meaning": (
                    "Fitted once on declared metadata, never on observed "
                    "values. Rebuilt deterministically at load time from the "
                    "checksummed level list — not unpickled, so a library "
                    "upgrade cannot silently change a stored fit."
                ),
                "families": [f["family"] for f in families if f["fit_scope"] == "global"],
                "persisted": False,
                "rebuilt_from": "categories_list.json + structural_fill sentinels",
            },
            "fold": {
                "meaning": (
                    "Fitted on training-fold rows only. Constructed by a "
                    "factory that requires the training index, and refuses to "
                    "serialize, so fold state cannot outlive its fold."
                ),
                "families": [f["family"] for f in families if f["fit_scope"] == "fold"],
                "persisted": False,
                "members": [
                    "StandardScaler (this ticket)",
                    "pairwise correlation prune (#12)",
                    "feature ranking (#12)",
                    "PCA(0.95) (#13)",
                ],
            },
        },
        "families": families,
        "levels": {
            "source": "categories_list.json",
            "declared_columns": len(declared),
            "sentinel_fills": sentinels,
            "sentinel_note": (
                "The sentinel is appended to the declared level list the "
                "pipeline builds (kb/column_spec.json -> categorical_levels). "
                "categories_list.json is given input and is never edited — "
                "extending the dataset is out of scope on the map."
            ),
        },
        "missing_values": {
            "structural": cspec.structural_fill,
            "after_fill": "zero nulls across all feature columns, asserted below",
        },
        "counts": {
            "boolean": by_name["boolean"]["n"],
            "categorical": by_name["categorical"]["n"],
            "cyclic": by_name["cyclic"]["n"],
            "linear_numeric": by_name["linear_numeric"]["n"],
            "model_input_width": width,
        },
    }


def assert_spec(spec: dict, cspec: columns.ColumnSpec, aspec: angles.AngularSpec) -> pd.DataFrame:
    """Check the contract against the source data. Raises on any drift."""
    frame = columns.apply_structural_fills(
        pd.read_csv(CSV_PATH, parse_dates=[cspec.key]), cspec
    )
    out = angles.transform(frame, aspec)

    counts = spec["counts"]
    assert counts["model_input_width"] == aspec.model_input_width, (
        f"width {counts['model_input_width']} disagrees with angular_spec "
        f"{aspec.model_input_width}"
    )

    # Every family's columns must exist in the transformed frame, exactly once,
    # and together they must account for the whole model input.
    seen: list[str] = []
    for family in spec["families"]:
        missing = [c for c in family["columns"] if c not in out.columns]
        assert not missing, f"{family['family']}: missing from frame: {missing[:5]}"
        seen.extend(family["columns"])
    assert len(seen) == len(set(seen)), "a column appears in two families"

    survivors = {
        c
        for c in out.columns
        if c not in {cspec.key, cspec.target_source, *cspec.dropped}
    }
    assert set(seen) == survivors, (
        f"families cover {len(seen)} columns, frame carries {len(survivors)} "
        f"— uncovered: {sorted(survivors - set(seen))[:5]}"
    )

    # Fit scope is the load-bearing claim: exactly one family may be fold-scoped,
    # and it must be the one with unbounded units.
    fold_families = [f["family"] for f in spec["families"] if f["fit_scope"] == "fold"]
    assert fold_families == ["linear_numeric"], (
        f"unexpected fold-scoped families: {fold_families}"
    )
    for family in spec["families"]:
        assert family["fit_scope"] in SCOPES, family["fit_scope"]

    # Cyclic columns must already be standardised by construction.
    cyclic = [f for f in spec["families"] if f["family"] == "cyclic"][0]["columns"]
    block = out[cyclic].to_numpy(dtype=float)
    assert np.isfinite(block).all(), "cyclic block carries non-finite values"
    assert block.min() >= -1.0 - 1e-9 and block.max() <= 1.0 + 1e-9, (
        f"cyclic block escapes [-1, 1]: [{block.min()}, {block.max()}]"
    )

    # Declared levels must cover everything observed, sentinel included.
    declared = json.loads(CATEGORIES_PATH.read_text())
    for col in cspec.categorical:
        levels = set(cspec.categorical_levels[col])
        observed = set(out[col].dropna().unique())
        unknown = observed - levels
        assert not unknown, f"{col}: observed levels not declared: {sorted(unknown)[:5]}"
        if col in cspec.structural_fill:
            sentinel = cspec.structural_fill[col]["value"]
            assert sentinel in levels, f"{col}: sentinel {sentinel!r} not in declared levels"
            assert sentinel not in set(declared.get(col, [])), (
                f"{col}: sentinel leaked into categories_list.json — that file is given input"
            )

    # The source file must be untouched.
    assert spec["source"]["categories_list"]["sha256"] == _sha256(CATEGORIES_PATH)

    features = list(seen)
    nulls = int(out[features].isna().sum().sum())
    assert nulls == 0, f"{nulls} nulls remain across feature columns after structural fill"

    return out


def render_md(spec: dict) -> str:
    counts = spec["counts"]
    lines = [
        "# Encoding spec — two views from one source",
        "",
        "Generated by `scripts/build_encoding_spec.py`. Do not hand-edit — rerun the script.",
        "",
        "Resolves [#9](https://github.com/jenujari/prdict-pov-v1/issues/9).",
        "",
        "**Scope.** " + spec["scope_note"],
        "",
        "## The contract",
        "",
        "| Family | n | Fit scope | Transform | XGBoost dtype | TFT dtype |",
        "|--------|---|-----------|-----------|---------------|-----------|",
    ]
    for f in spec["families"]:
        lines.append(
            f"| `{f['family']}` | {f['n']} | **{f['fit_scope']}** | {f['transform']} | "
            f"{f['dtype']['xgboost']} | {f['dtype']['tft']} |"
        )
    lines += [
        f"| **total** | **{counts['model_input_width']}** | | | | |",
        "",
        "Both views carry the same columns and differ only in dtype and in how "
        "categoricals are presented. There is one encoded source of truth, not two "
        "divergent feature sets.",
        "",
        "## Fit scope is the whole point",
        "",
        "Three scopes, and the difference between them is where leakage would enter:",
        "",
    ]
    for name in SCOPES:
        s = spec["fit_scopes"][name]
        fams = ", ".join(f"`{x}`" for x in s["families"]) or "—"
        lines.append(f"- **`{name}`** — {s['meaning']} Families: {fams}.")
    lines += [
        "",
        "The asymmetry that makes this worth enforcing: the label encoders are "
        "fitted **once and shared by every fold**, while the scaler must be "
        "refitted **inside every fold**. Both are 'fitted objects' and the "
        "correct handling is opposite, so `prdict/encoding.py` splits them by "
        "construction rather than by convention:",
        "",
        "```python",
        "load_global(spec)        # no train argument exists to pass",
        "build_fold(spec, train_idx)   # train index is required, result will not pickle",
        "```",
        "",
        "`load_global()` **rebuilds** the encoders from the checksummed level "
        "list rather than unpickling a stored fit — they are a pure function of "
        "`categories_list.json`, so there is nothing to persist and no stored "
        "fit that a library upgrade could silently reinterpret.",
        "",
        "Fold-scoped members, in fit order: "
        + ", ".join(spec["fit_scopes"]["fold"]["members"])
        + ".",
        "",
        "## Categorical levels",
        "",
        f"`categories_list.json` declares levels for {spec['levels']['declared_columns']} "
        f"columns; {counts['categorical']} of them are typed categorical by #3. "
        "Levels come from that declared list, never from observed values, so an "
        "unobserved level still gets a code and an embedding row.",
        "",
        "**The source file is never edited.** " + spec["levels"]["sentinel_note"],
        "",
    ]
    for col, value in spec["levels"]["sentinel_fills"].items():
        rule = spec["missing_values"]["structural"][col]
        lines.append(
            f"- `{col}` — filled with `\"{value}\"` when `{rule['condition']}`. "
            f"{rule['reason']}."
        )
    lines += [
        "",
        "After the structural fill, **zero nulls** remain across all "
        f"{counts['model_input_width']} feature columns. The build asserts it.",
        "",
        "## What is not transformed",
        "",
        "- **Skew handling is dropped entirely.** `backup.one.txt` applied `log1p` "
        "wherever `|skew| > 1.5`; the signed speed columns go negative, so that "
        "yields NaN. A scaler already gives the TFT the conditioning it needs, and "
        "XGBoost is invariant to monotone rescaling of a split variable.",
        "- **Cyclic pairs are not scaled.** Standardising the two halves separately "
        "breaks the unit-circle identity that makes the pair an angle.",
        "- **`_dist` is not put on a circle**, and is not dropped either — it rides "
        "in `linear_numeric` beside the recomputed `_sep` pairs, and #12 chooses. "
        "See `kb/angular_spec.md`.",
        "",
        "## Checked on every run",
        "",
        "- Family counts agree with `column_spec.json` and `angular_spec.json`, and "
        f"sum to the model input width `angular_spec` reports ({counts['model_input_width']}).",
        "- Families are disjoint and cover every surviving column — no column "
        "encoded twice, none left unencoded.",
        "- Exactly one family is fold-scoped.",
        "- The cyclic block is finite and inside `[-1, 1]`.",
        "- Every observed level is declared; every sentinel is in the built list and "
        "**absent from `categories_list.json`**.",
        "- `categories_list.json` matches its recorded checksum.",
        "- Zero feature nulls after the structural fill.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cspec = columns.load_spec()
    aspec = angles.load_spec()

    spec = build_spec(cspec, aspec)
    out = assert_spec(spec, cspec, aspec)

    OUT_JSON.write_text(json.dumps(spec, indent=1) + "\n")
    OUT_MD.write_text(render_md(spec))

    counts = spec["counts"]
    print(f"frame after angular transform: {out.shape[1]} columns x {out.shape[0]} rows")
    for family in spec["families"]:
        print(
            f"  {family['family']:16s}: {family['n']:3d}  "
            f"fit_scope={family['fit_scope']}"
        )
    print(f"  {'TOTAL':16s}: {counts['model_input_width']:3d}")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
