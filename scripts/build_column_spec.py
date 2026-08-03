"""Derive the column specification for nft50.csv.

Resolves wayfinder ticket #3. Every one of the 240 columns is assigned exactly
one role, and every feature column exactly one type, by rule rather than by
hand-maintained list — rerun this after any change to the source data.

Writes kb/column_spec.json (machine-readable) and kb/column_spec.md (human).

    uv run python scripts/build_column_spec.py
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "nft50.csv"
CATEGORIES_PATH = ROOT / "categories_list.json"
SPEC_JSON = ROOT / "kb" / "column_spec.json"
SPEC_MD = ROOT / "kb" / "column_spec.md"

KEY = "record_date"
TARGET_SOURCE = "c"
DROPPED_PRICE = ["o", "h", "l"]

# Planets present in the ephemeris, in the order the CSV lays them out.
PLANETS = [
    "sun",
    "moon",
    "saturn",
    "venus",
    "mercury",
    "jupiter",
    "rahu",
    "ketu",
    "mars",
]

# Cross-planet angular separations are named "<a>_<b>_dist". The per-planet
# "<p>_speed_dist" columns also end in _dist but are rates of change, not
# angles — they must not be swept up by the same rule.
CROSS_DIST_RE = re.compile(rf"^({'|'.join(PLANETS)})_({'|'.join(PLANETS)})_dist$")

# Why each constant column is constant. Every one of these is structural — a
# consequence of astronomy or of the bala formulae — not an artifact of this
# particular 27-year sample. That distinction is load-bearing: it licenses
# dropping them once, globally, instead of re-deciding inside every CV fold.
CONSTANT_REASONS = {
    "sun_is_retro": "The Sun never appears retrograde from Earth.",
    "moon_is_retro": "The Moon never appears retrograde from Earth.",
    "rahu_is_retro": "Rahu is a computed lunar node; it is always retrograde by convention.",
    "ketu_is_retro": "Ketu is a computed lunar node; it is always retrograde by convention.",
    "rahu_speed_category": "Follows from rahu_is_retro — a permanently retrograde body is always 'vakra'.",
    "ketu_speed_category": "Follows from ketu_is_retro — a permanently retrograde body is always 'vakra'.",
    "rahu_vedha": "Vedha is not defined for the shadow planets; the source emits a fixed placeholder.",
    "ketu_vedha": "Vedha is not defined for the shadow planets; the source emits a fixed placeholder.",
    "rahu_distance": "The nodes are geometric points, not bodies — no physical distance exists.",
    "ketu_distance": "The nodes are geometric points, not bodies — no physical distance exists.",
    "rahu_ketu_dist": "Rahu and Ketu are antipodal by construction, so their separation is always 180 degrees.",
    "sun_uday_bala": "Formula constant for the Sun.",
    "rahu_uday_bala": "Formula constant for the nodes.",
    "ketu_uday_bala": "Formula constant for the nodes.",
    "sun_vakra_bala": "Retrograde strength is zero for a body that never retrogrades.",
    "moon_vakra_bala": "Retrograde strength is zero for a body that never retrogrades.",
    "rahu_vakra_bala": "Retrograde strength is maximal for a permanently retrograde body.",
    "ketu_vakra_bala": "Retrograde strength is maximal for a permanently retrograde body.",
    "rahu_sign_lordship": (
        "Declared with two levels but only 'Enemy' occurs. Rahu traverses all twelve "
        "signs roughly every 18.6 years, so 27 years of history covers the full cycle "
        "many times over — the unobserved level is unreachable, not merely unsampled."
    ),
    "ketu_sign_lordship": (
        "Declared with two levels but only 'Enemy' occurs. Same full-cycle-coverage "
        "argument as rahu_sign_lordship."
    ),
}

# Null in sun_vedha_target is structural: it is null exactly when sun_vedha is
# 'no', i.e. when no vedha is in effect there is no target nakshatra to name.
# Verified 1:1 against the data below. Filled with a sentinel level rather than
# imputed, and the sentinel is appended to the declared level list.
STRUCTURAL_FILL = {
    "sun_vedha_target": {
        "value": "none",
        "condition": "sun_vedha == 'no'",
        "reason": "structural — no vedha in effect means no target nakshatra exists",
    }
}

# Exact functional dependencies between feature columns. These are deterministic,
# not merely correlated, so the pairwise-|r| redundancy stage in #12 will not
# catch them — a nonlinear function of a difference of two columns has no strong
# linear correlation with either one.
DERIVED_RELATIONSHIPS = {
    "tithy": {
        "formula": "floor(((moon_longitude - sun_longitude) mod 360) / 12) + 1",
        "inputs": ["moon_longitude", "sun_longitude"],
        "note": (
            "Exact on 100% of rows. This is also why there is no sun_moon_dist column "
            "among the 35 cross-planet separations — tithy is that separation, binned "
            "into 30 steps of 12 degrees."
        ),
    }
}


def classify_type(col: str, frame: pd.DataFrame, category_keys: set[str]) -> str:
    """Assign exactly one type to a feature column.

    Order matters: bool dtype wins over the categories file, because 14 of the
    boolean columns are also declared there and would otherwise be double-typed.
    """
    if frame[col].dtype == bool:
        return "boolean"
    if col in category_keys:
        return "categorical"
    if col.endswith("_longitude"):
        return "angular"
    if CROSS_DIST_RE.match(col):
        return "angular"
    if col.endswith("_nakshatra_pada") or col == "tithy":
        return "angular"
    return "linear_numeric"


def build() -> dict:
    frame = pd.read_csv(CSV_PATH, parse_dates=[KEY])
    categories = json.loads(CATEGORIES_PATH.read_text())
    category_keys = set(categories)

    all_columns = list(frame.columns)

    # --- Roles -------------------------------------------------------------
    constant = sorted(
        c
        for c in all_columns
        if c not in (KEY, TARGET_SOURCE, *DROPPED_PRICE)
        and frame[c].nunique(dropna=True) <= 1
    )
    assigned = {KEY, TARGET_SOURCE, *DROPPED_PRICE, *constant}
    features = [c for c in all_columns if c not in assigned]

    roles = {
        "key": [KEY],
        "target_source": [TARGET_SOURCE],
        "dropped_price": DROPPED_PRICE,
        "dropped_constant": constant,
        "feature": features,
    }

    # --- Types -------------------------------------------------------------
    types: dict[str, list[str]] = {
        "boolean": [],
        "categorical": [],
        "angular": [],
        "linear_numeric": [],
    }
    for col in features:
        types[classify_type(col, frame, category_keys)].append(col)

    # --- Availability ------------------------------------------------------
    # Map decision 3 asks for a known-future vs past-only tag. With price
    # features dropped (decision 5), the expectation is that this comes out
    # uniformly known_future. Assert it rather than assume it.
    cutoff = frame.loc[frame[TARGET_SOURCE].notna(), KEY].max()
    future = frame[KEY] > cutoff
    unavailable = {
        col: int(frame.loc[future, col].isna().sum())
        for col in features
        if frame.loc[future, col].isna().any()
    }
    # A column whose only future nulls are structural still counts as known.
    genuinely_unavailable = {
        col: n for col, n in unavailable.items() if col not in STRUCTURAL_FILL
    }

    # --- Declared levels, with sentinels folded in -------------------------
    levels = {}
    for col in types["categorical"]:
        declared = [str(v) for v in categories[col]]
        if col in STRUCTURAL_FILL:
            declared = declared + [STRUCTURAL_FILL[col]["value"]]
        levels[col] = declared

    never_observed = {}
    for col, declared in levels.items():
        observed = set(frame[col].dropna().astype(str).unique())
        if col in STRUCTURAL_FILL:
            observed.add(STRUCTURAL_FILL[col]["value"])
        missing = [lv for lv in declared if lv not in observed]
        if missing:
            never_observed[col] = missing

    # log1p is unsafe wherever a column reaches -1 or below; backup.one.txt
    # applied it to anything with |skew| > 1.5, which silently produced NaN.
    log1p_unsafe = sorted(
        col
        for col in types["linear_numeric"] + types["angular"]
        if pd.api.types.is_numeric_dtype(frame[col]) and float(frame[col].min()) <= -1
    )

    spec = {
        "generated": str(date.today()),
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/3",
        "source": {
            "csv": CSV_PATH.name,
            "categories": CATEGORIES_PATH.name,
            "n_columns": len(all_columns),
            "n_rows": int(len(frame)),
            "label_cutoff": str(cutoff.date()),
        },
        "roles": roles,
        "types": types,
        "availability": {
            "known_future": features,
            "past_only": sorted(genuinely_unavailable),
        },
        "categorical_levels": levels,
        "never_observed_levels": never_observed,
        "structural_fill": STRUCTURAL_FILL,
        "derived_relationships": DERIVED_RELATIONSHIPS,
        "constant_reasons": CONSTANT_REASONS,
        "log1p_unsafe": log1p_unsafe,
    }

    validate(spec, frame)
    return spec


def validate(spec: dict, frame: pd.DataFrame) -> None:
    """Every column accounted for exactly once, at both role and type level."""
    all_columns = list(frame.columns)

    role_members = [c for members in spec["roles"].values() for c in members]
    assert len(role_members) == len(set(role_members)), "a column carries two roles"
    assert set(role_members) == set(all_columns), (
        f"role coverage gap: {set(all_columns) ^ set(role_members)}"
    )
    assert len(role_members) == spec["source"]["n_columns"]

    type_members = [c for members in spec["types"].values() for c in members]
    assert len(type_members) == len(set(type_members)), "a column carries two types"
    assert set(type_members) == set(spec["roles"]["feature"]), (
        f"type coverage gap: {set(spec['roles']['feature']) ^ set(type_members)}"
    )

    # Map decision 3: with price features dropped, nothing should be past-only.
    assert not spec["availability"]["past_only"], (
        f"unexpected past-only columns: {spec['availability']['past_only']}"
    )

    # The structural fill claim must actually hold in the data.
    for col, rule in spec["structural_fill"].items():
        driver, _, value = rule["condition"].partition(" == ")
        expected = frame[driver.strip()] == value.strip().strip("'")
        assert (frame[col].isna() == expected).all(), (
            f"{col} nulls do not match {rule['condition']} exactly"
        )

    # Every constant column must have a documented reason.
    undocumented = set(spec["roles"]["dropped_constant"]) - set(spec["constant_reasons"])
    assert not undocumented, f"constant columns lack a reason: {sorted(undocumented)}"

    # A claimed exact dependency has to actually be exact.
    elongation = (frame["moon_longitude"] - frame["sun_longitude"]) % 360
    assert ((elongation // 12) + 1 == frame["tithy"]).all(), (
        "tithy is not exactly floor(elongation / 12) + 1"
    )


def render_markdown(spec: dict) -> str:
    roles, types = spec["roles"], spec["types"]
    src = spec["source"]

    def bullets(cols: list[str]) -> str:
        return "\n".join(f"- `{c}`" for c in cols)

    lines = [
        "# Column specification — `nft50.csv`",
        "",
        f"Generated {spec['generated']} by `scripts/build_column_spec.py`. "
        "Do not hand-edit — rerun the script.",
        "",
        f"Resolves [#3]({spec['resolves']}).",
        "",
        f"`{src['csv']}`: **{src['n_columns']} columns**, {src['n_rows']} rows. "
        f"Labels (`c`) run to **{src['label_cutoff']}**.",
        "",
        "## Roles",
        "",
        "| Role | Count | Meaning |",
        "|------|-------|---------|",
        f"| `key` | {len(roles['key'])} | The date index. |",
        f"| `target_source` | {len(roles['target_source'])} | Close price — source of the label, never an input (map decision 5). |",
        f"| `dropped_price` | {len(roles['dropped_price'])} | Open/high/low, dropped with all price-derived features (map decision 5). |",
        f"| `dropped_constant` | {len(roles['dropped_constant'])} | Single-valued across the file. All structural — see below. |",
        f"| `feature` | {len(roles['feature'])} | Model inputs. |",
        "",
        "## Types",
        "",
        "| Type | Count |",
        "|------|-------|",
        f"| `boolean` | {len(types['boolean'])} |",
        f"| `categorical` | {len(types['categorical'])} |",
        f"| `angular` | {len(types['angular'])} |",
        f"| `linear_numeric` | {len(types['linear_numeric'])} |",
        f"| **total** | **{len(roles['feature'])}** |",
        "",
        "### `boolean`",
        "",
        f"{len(types['boolean'])} columns, all `*_is_retro` or `*_vargottama`. "
        "Note these are also declared in `categories_list.json`; the bool dtype wins, "
        "so they are typed once, here.",
        "",
        bullets(types["boolean"]),
        "",
        "### `categorical`",
        "",
        f"{len(types['categorical'])} columns. Levels come from `categories_list.json` — "
        "the **full declared list**, never the observed values, so that encoding is stable "
        "across CV folds (map decision 6).",
        "",
        bullets(types["categorical"]),
        "",
        "### `angular`",
        "",
        f"{len(types['angular'])} columns. This spec fixes **membership only**. "
        "Each column's period and its sin/cos transform are decided in "
        "[#7](https://github.com/jenujari/prdict-pov-v1/issues/7).",
        "",
        bullets(types["angular"]),
        "",
        "### `linear_numeric`",
        "",
        f"{len(types['linear_numeric'])} columns — latitudes (a narrow ±8.6 band, not cyclic), "
        "distances in AU, the three speed families, and the bala scores.",
        "",
        bullets(types["linear_numeric"]),
        "",
        "## Availability: known-future vs past-only",
        "",
        "Map decision 3 called for splitting covariates into known-future and past-only. "
        "With price features dropped (decision 5), **every one of the "
        f"{len(roles['feature'])} feature columns is `known_future`** — the ephemeris is "
        "computed, so it is fully populated through the end of the file. The build script "
        "asserts this rather than assuming it.",
        "",
        "The practical consequence: the past-60 and future-30 blocks carry an **identical "
        "feature set**, so the 90-step window is simply contiguous and the encoder/decoder "
        "boundary is a slicing convention. This is the input "
        "[#11](https://github.com/jenujari/prdict-pov-v1/issues/11) was told to verify.",
        "",
        "## Structural nulls",
        "",
        "`sun_vedha_target` is the only feature column with missing values (2685 rows, "
        "55 of them after the label cutoff). The nulls are **structural, not missing**: "
        "the column is null exactly when `sun_vedha == 'no'`, a perfect 1:1 match that the "
        "build script asserts. No vedha in effect means there is no target nakshatra to name.",
        "",
        "Handled by appending a `\"none\"` sentinel to the declared level list and filling "
        "with it — not by imputation, which would invent a nakshatra that astronomically "
        "is not there.",
        "",
        "## Constant columns",
        "",
        f"All {len(roles['dropped_constant'])} constant columns are constant **by definition**, "
        "not by sampling accident. That is what licenses dropping them once, globally, rather "
        "than re-deciding inside every CV fold — the concern raised in "
        "[#10](https://github.com/jenujari/prdict-pov-v1/issues/10).",
        "",
        "| Column | Why constant |",
        "|--------|--------------|",
    ]
    for col in roles["dropped_constant"]:
        lines.append(f"| `{col}` | {spec['constant_reasons'][col]} |")

    lines += [
        "",
        "## Exact derived relationships",
        "",
        "Deterministic dependencies between feature columns. These matter because the "
        "stage-1 redundancy prune in [#12](https://github.com/jenujari/prdict-pov-v1/issues/12) "
        "works on pairwise `|r|` and **will not catch them** — a nonlinear function of a "
        "difference of two columns has no strong linear correlation with either input.",
        "",
    ]
    for col, rel in spec["derived_relationships"].items():
        lines += [
            f"### `{col}`",
            "",
            f"```\n{col} = {rel['formula']}\n```",
            "",
            rel["note"],
            "",
        ]

    lines += [
        "## Declared levels never observed",
        "",
        "These levels appear in `categories_list.json` but never in 27 years of data. They are "
        "retained in the encoding anyway (map decision 6) so that category codes stay stable "
        "and an unseen level at inference time does not shift every other code.",
        "",
        "| Column | Never observed |",
        "|--------|----------------|",
    ]
    for col, missing in spec["never_observed_levels"].items():
        lines.append(f"| `{col}` | {', '.join(f'`{m}`' for m in missing)} |")

    lines += [
        "",
        "## `log1p` is unsafe here",
        "",
        "`backup.one.txt` applied `log1p` to any feature with `|skew| > 1.5`. The following "
        "columns reach `-1` or below, where `log1p` returns NaN:",
        "",
        bullets(spec["log1p_unsafe"]),
        "",
        "Signed rate-of-change columns must not be log-transformed. Carried into "
        "[#9](https://github.com/jenujari/prdict-pov-v1/issues/9), which decides skew handling.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    spec = build()
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    SPEC_MD.write_text(render_markdown(spec))

    roles, types = spec["roles"], spec["types"]
    print(f"columns          : {spec['source']['n_columns']}")
    for role, members in roles.items():
        print(f"  {role:18s}: {len(members)}")
    print("feature types")
    for typ, members in types.items():
        print(f"  {typ:18s}: {len(members)}")
    print(f"\nwrote {SPEC_JSON.relative_to(ROOT)} and {SPEC_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
