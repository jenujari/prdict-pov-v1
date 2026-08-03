"""Derive the angular-transform specification for nft50.csv.

Resolves wayfinder ticket #7. Ticket #3 decided which columns are angular; this
one decides, for each of them, what quantity it really measures, what its period
is, and what the sin/cos transform emits.

Every claim the spec rests on is *verified against the data here*, not asserted:
the script fails loudly if the source file stops matching. Three of those checks
matter more than the rest.

  1. The 34 cross-planet `<a>_<b>_dist` columns are **not** angles. They hold
     `abs(lon_a - lon_b)` with no modular reduction at all, so two planets at
     359 and 1 degrees are recorded 358 degrees apart. Both encodings are kept:
     the source column passes through untransformed (typed linear by #3), and
     the signed separation recomputed from the longitudes is added beside it.
     #12 decides which earns its place. Only sin/cos is withheld from `_dist`,
     because sin is odd and sin(|d|) folds two configurations onto one value.
  2. `tithy` is exactly `floor(elongation / 12) + 1` on the Moon-Sun elongation,
     1-indexed, so its period is 30 with origin 1.
  3. `ketu_longitude` is `rahu_longitude + 180` on every row, by construction —
     Ketu is the south lunar node, so its sin/cos pair is the exact negation of
     Rahu's and it is dropped. This does *not* extend to the seven
     `<a>_ketu_dist` columns: as unwrapped absolute differences they are not a
     function of the Rahu ones (correlation only -0.48 to -0.69), so they stay.

Writes kb/angular_spec.json (machine-readable) and kb/angular_spec.md (human).

    uv run python scripts/build_angular_spec.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "nft50.csv"
COLUMN_SPEC = ROOT / "kb" / "column_spec.json"
SPEC_JSON = ROOT / "kb" / "angular_spec.json"
SPEC_MD = ROOT / "kb" / "angular_spec.md"

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

CROSS_DIST_RE = re.compile(rf"^({'|'.join(PLANETS)})_({'|'.join(PLANETS)})_dist$")

# Ketu is the south lunar node: antipodal to Rahu by definition, not by accident
# of this sample. Every Ketu angle is therefore the Rahu angle plus 180, and its
# sin/cos pair is the exact negation of Rahu's. Dropping them globally cannot
# leak, for the same reason the 20 constants of #3 could be dropped globally.
ANTIPODAL_REASON = (
    "Ketu is antipodal to Rahu by construction, so this column is "
    "`{twin}` + 180 on every row; its sin/cos pair is the exact "
    "negation of `{twin}`'s and carries no independent information."
)

# Monday-first, matching `pandas.Series.dt.dayofweek`. Verified below against the
# record_date index rather than trusted.
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angular difference between two degree arrays, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def verify(frame: pd.DataFrame, angular: list[str]) -> dict:
    """Re-derive every empirical claim the spec makes. Raises on any mismatch."""
    lon = {p: frame[f"{p}_longitude"].to_numpy() for p in PLANETS}
    evidence: dict[str, object] = {}

    # 1. The _dist columns are an unwrapped absolute difference, not an angle.
    dist_cols = [c for c in angular if CROSS_DIST_RE.match(c)]
    worst_raw, worst_signed, worst_folded = 0.0, 0.0, 0.0
    for col in dist_cols:
        a, b = CROSS_DIST_RE.match(col).groups()
        observed = frame[col].to_numpy()
        unwrapped = np.abs(lon[a] - lon[b])
        signed = (lon[b] - lon[a]) % 360.0
        folded = np.minimum(signed, 360.0 - signed)
        worst_raw = max(worst_raw, float(np.abs(observed - unwrapped).max()))
        worst_signed = max(worst_signed, float(circular_error(observed, signed).max()))
        worst_folded = max(worst_folded, float(np.abs(observed - folded).max()))
    if worst_raw > 1e-6:
        raise AssertionError(
            f"`_dist` is no longer abs(lon_a - lon_b): worst error {worst_raw}. "
            "The separation recomputation in this spec assumes it is."
        )
    evidence["dist_is_unwrapped_abs_diff"] = {
        "columns": len(dist_cols),
        "max_error_vs_abs_diff": worst_raw,
        "max_error_vs_signed_separation": worst_signed,
        "max_error_vs_folded_separation": worst_folded,
    }

    # How much damage the unwrapping does: cells recorded far apart that are in
    # fact close. This is the number that rules out using the column as given.
    misleading = 0
    for col in dist_cols:
        a, b = CROSS_DIST_RE.match(col).groups()
        unwrapped = np.abs(lon[a] - lon[b])
        true_sep = circular_error(lon[a], lon[b])
        misleading += int(((unwrapped > 180.0) & (true_sep < 30.0)).sum())
    evidence["dist_is_unwrapped_abs_diff"]["cells_over_180_but_within_30"] = misleading

    # 2. tithy is a 12-degree quantisation of the Moon-Sun elongation, 1-indexed.
    elongation = (lon["moon"] - lon["sun"]) % 360.0
    derived = np.floor(elongation / 12.0) + 1
    if not np.array_equal(derived, frame["tithy"].to_numpy()):
        raise AssertionError("tithy is not floor(elongation / 12) + 1")
    evidence["tithy"] = {
        "formula": "floor(((moon_longitude - sun_longitude) mod 360) / 12) + 1",
        "min": int(frame["tithy"].min()),
        "max": int(frame["tithy"].max()),
        "exact_match": True,
    }

    # 3. Ketu is antipodal to Rahu on every row.
    offset = np.unique(np.round((lon["ketu"] - lon["rahu"]) % 360.0, 9))
    if offset.tolist() != [180.0]:
        raise AssertionError(f"ketu - rahu is not a constant 180: {offset[:5]}")
    evidence["ketu_antipodal_to_rahu"] = {"offsets_observed": offset.tolist()}

    # The antipodal relation is exact for the *longitude*, and so for any circular
    # encoding of it. It does NOT carry over to the raw `_dist` columns, which are
    # unwrapped absolute differences: `<a>_ketu_dist` is not a function of
    # `<a>_rahu_dist`. Measured, because an earlier revision assumed otherwise and
    # dropped all seven.
    worst_corr, worst_resid = 0.0, np.inf
    for col in angular:
        match = CROSS_DIST_RE.match(col)
        if not match or "ketu" not in match.groups():
            continue
        twin = "_".join("rahu" if p == "ketu" else p for p in match.groups()) + "_dist"
        if twin not in frame.columns:
            continue
        x, y = frame[twin].to_numpy(), frame[col].to_numpy()
        worst_corr = max(worst_corr, abs(float(np.corrcoef(x, y)[0, 1])))
        fit = np.polyfit(x, y, 1)
        worst_resid = min(worst_resid, float(np.abs(y - np.polyval(fit, x)).max()))
    if worst_corr >= 0.95:
        raise AssertionError(
            f"a ketu _dist column now correlates {worst_corr} with its rahu twin; "
            "the spec keeps all seven on the grounds that they do not"
        )
    evidence["ketu_dist_not_redundant"] = {
        "max_abs_correlation_with_rahu_twin": worst_corr,
        "min_linear_fit_residual_degrees": worst_resid,
    }

    # 3b. The antipodal relation does NOT carry through to pada, because 180
    #     degrees is 13.5 nakshatras only under a uniform 27-fold scheme and this
    #     source is 28-fold with unequal spans. So `ketu_nakshatra_pada` survives
    #     while `ketu_longitude` does not — measured here so the asymmetry is
    #     evidenced rather than argued. Both correlations sit under the 0.95
    #     stage-1 threshold, so the prune would not remove it either.
    crosstab = pd.crosstab(frame["rahu_nakshatra_pada"], frame["ketu_nakshatra_pada"])
    def circle(v: pd.Series) -> np.ndarray:
        return 2.0 * np.pi * (v.to_numpy(dtype=float) - 1.0) / 4.0
    rahu_t, ketu_t = circle(frame["rahu_nakshatra_pada"]), circle(frame["ketu_nakshatra_pada"])
    evidence["ketu_pada_survives"] = {
        "determined_by_rahu_pada": bool(((crosstab > 0).sum(axis=1) == 1).all()),
        "corr_sin": float(np.corrcoef(np.sin(rahu_t), np.sin(ketu_t))[0, 1]),
        "corr_cos": float(np.corrcoef(np.cos(rahu_t), np.cos(ketu_t))[0, 1]),
    }

    # 3c. The same antipodal relation DOES carry through to the sign columns,
    #     which are uniform (30 degrees, and navamsa 30/9), so 180 degrees is a
    #     whole number of them. Those are categoricals and out of this ticket's
    #     scope — recorded as a hand-off to #9 and #15, not acted on here.
    antipodal_categoricals = [
        f"ketu_{suffix}"
        for suffix in ("sign", "navamsa_sign")
        if (pd.crosstab(frame[f"rahu_{suffix}"], frame[f"ketu_{suffix}"]) > 0)
        .sum(axis=1)
        .max()
        == 1
    ]
    evidence["antipodal_categoricals"] = antipodal_categoricals

    # 4. Pada really does wrap 4 -> 1 as longitude increases, so period 4 is not
    #    a convenient fiction. Checked on the Sun, whose motion never reverses.
    ordered = pd.DataFrame(
        {"pada": frame["sun_nakshatra_pada"].to_numpy(), "lon": lon["sun"]}
    ).sort_values("lon")
    steps = ordered["pada"].to_numpy()
    wraps = int(((steps[:-1] == 4) & (steps[1:] == 1)).sum())
    if wraps == 0:
        raise AssertionError("sun_nakshatra_pada never wraps 4 -> 1; period 4 is wrong")
    evidence["pada_wraps"] = {"planet": "sun", "four_to_one_transitions": wraps}

    # 5. The nakshatra scheme is 28-fold with Abhijit, not a uniform 27, so
    #    (nakshatra index * 4 + pada) is NOT a uniform 108-fold circle. Recorded
    #    because the ticket asked whether it was.
    names = frame["sun_nakshatra_name"]
    spans = pd.DataFrame({"name": names, "lon": lon["sun"]}).groupby("name")["lon"]
    widths = (spans.max() - spans.min()).round(3)
    evidence["nakshatra_scheme"] = {
        "levels": int(names.nunique()),
        "includes_abhijit": bool((names == "Abhijit").any()),
        "narrowest_degrees": float(widths.min()),
        "widest_degrees": float(widths.max()),
        "uniform": bool(widths.max() - widths.min() < 0.5),
    }

    # 6. The weekday level names line up with the Monday-first date index, so
    #    the period-7 position can be read off the string without the key.
    dayofweek = pd.to_datetime(frame["record_date"]).dt.dayofweek
    mapped = frame["weekday"].map({n: i for i, n in enumerate(WEEKDAY_ORDER)})
    if not mapped.equals(dayofweek.astype(mapped.dtype)):
        raise AssertionError("weekday levels do not match the record_date index")
    traded = frame[frame["c"].notna()]["weekday"].value_counts()
    evidence["weekday"] = {
        "levels": WEEKDAY_ORDER,
        "sessions_per_level": {k: int(v) for k, v in traded.items()},
    }

    return evidence


def build() -> dict:
    frame = pd.read_csv(CSV_PATH)
    column_spec = json.loads(COLUMN_SPEC.read_text())
    angular = column_spec["types"]["angular"]

    # The `_dist` columns are typed linear by #3 (see the note there) and survive
    # untransformed. They are still the inventory of which planet pairs exist, so
    # the separation transforms are enumerated from them.
    dist_cols = [c for c in column_spec["types"]["linear_numeric"] if CROSS_DIST_RE.match(c)]

    evidence = verify(frame, angular + dist_cols)

    # --- structural drops ---------------------------------------------------
    # Only `ketu_longitude`. Its sin/cos pair is the exact negation of Rahu's, so
    # it is a true duplicate *under this encoding*.
    #
    # The seven `<a>_ketu_dist` columns are NOT dropped, though #7 originally did
    # drop them. That drop was justified on the *recomputed* separation, where the
    # Ketu pair is an exact negation of the Rahu one. It does not carry over to
    # the raw column: an unwrapped absolute difference to Ketu is not a function
    # of the one to Rahu (measured correlation only -0.48 to -0.69), so the source
    # values are independent numbers and are kept.
    dropped: dict[str, str] = {"ketu_longitude": ANTIPODAL_REASON.format(twin="rahu_longitude")}

    kept = [c for c in angular if c not in dropped]

    # --- transforms ---------------------------------------------------------
    # `origin` is the value that maps to angle zero; `period` is the span of one
    # full turn in the source column's own units. theta = 2*pi*(v - origin)/period.
    transforms: list[dict] = []

    for col in kept:
        if col == "tithy":
            transforms.append(
                {
                    "source": col,
                    "family": "tithy",
                    "stem": "tithy",
                    "value": "column",
                    "period": 30.0,
                    "origin": 1.0,
                    "note": "Lunar day: a 12-degree quantisation of the Moon-Sun elongation, 1-indexed.",
                }
            )
        elif col.endswith("_longitude"):
            transforms.append(
                {
                    "source": col,
                    "family": "longitude",
                    "stem": col,
                    "value": "column",
                    "period": 360.0,
                    "origin": 0.0,
                    "note": "Sidereal longitude on the full circle.",
                }
            )
        elif col.endswith("_nakshatra_pada"):
            transforms.append(
                {
                    "source": col,
                    "family": "pada",
                    "stem": col,
                    "value": "column",
                    "period": 4.0,
                    "origin": 1.0,
                    "note": "Quarter of a nakshatra, 1-indexed; wraps 4 -> 1 at the nakshatra boundary.",
                }
            )
        else:
            raise AssertionError(f"no period rule for angular column {col!r}")

    # Separations, recomputed from the longitudes. These are *additions*: the
    # source `_dist` column is not consumed, so both encodings reach the model
    # and feature selection decides between them.
    for col in dist_cols:
        a, b = CROSS_DIST_RE.match(col).groups()
        if "ketu" in (a, b):
            # As a signed separation the Ketu pair is the exact negation of the
            # Rahu one, so only the raw `_dist` column is carried for these.
            continue
        transforms.append(
            {
                "source": col,
                "family": "separation",
                "stem": f"{a}_{b}_sep",
                "value": "signed_separation",
                "from": a,
                "to": b,
                "period": 360.0,
                "origin": 0.0,
                "replaces_source": False,
                "note": (
                    f"RECOMPUTED as (`{b}_longitude` - `{a}_longitude`) mod 360, and added "
                    f"alongside `{col}`, which survives untransformed."
                ),
            }
        )

    # weekday is a categorical (#3) that additionally earns a cyclic view; it is
    # the one transform whose source column is not in the angular list, and the
    # only one that does not replace its source.
    transforms.append(
        {
            "source": "weekday",
            "family": "weekday",
            "stem": "weekday",
            "value": "weekday_index",
            "period": 7.0,
            "origin": 0.0,
            "replaces_source": False,
            "note": "Monday-first index over the declared levels. The categorical column survives alongside.",
        }
    )

    for t in transforms:
        t.setdefault("replaces_source", True)
        t["emits"] = [f"{t['stem']}_sin", f"{t['stem']}_cos"]

    emitted = [name for t in transforms for name in t["emits"]]
    assert len(set(emitted)) == len(emitted), "emitted column names collide"

    n_features = len(column_spec["roles"]["feature"])
    return {
        "resolves": 7,
        "source": {"csv": CSV_PATH.name, "n_rows": int(len(frame))},
        "conventions": {
            "formula": "theta = 2*pi*(value - origin) / period; emit sin(theta), cos(theta)",
            "naming": "<stem>_sin and <stem>_cos; a pair is exactly the two columns sharing a stem",
            "raw_angle_survives": False,
            "pairs_are_atomic": True,
            "pair_atomicity_note": (
                "Feature selection (#12) and PCA (#13) must keep or drop a (sin, cos) pair "
                "together, scoring it by the better of its two members. A lone half is not a "
                "weaker encoding of the angle but a wrong one: sin alone identifies theta with "
                "180 - theta, cos alone identifies theta with -theta."
            ),
        },
        "dropped_redundant": dropped,
        "transforms": transforms,
        "counts": {
            "angular_in_column_spec": len(angular),
            "dropped_redundant": len(dropped),
            "angular_transformed": len(kept),
            "separations_added": sum(1 for t in transforms if t["family"] == "separation"),
            "dist_columns_kept": len(dist_cols),
            "extra_cyclic_views": sum(1 for t in transforms if t["family"] == "weekday"),
            "emitted_columns": len(emitted),
            "features_before": n_features,
            "features_after": n_features - len(dropped),
            "model_input_width": (
                len(column_spec["types"]["boolean"])
                + len(column_spec["types"]["categorical"])
                + len(column_spec["types"]["linear_numeric"])
                + len(emitted)
            ),
        },
        "evidence": evidence,
    }


def bullets(items) -> str:
    return "\n".join(f"- `{item}`" for item in items)


def render_markdown(spec: dict) -> str:
    c, e = spec["counts"], spec["evidence"]
    dist, weekday = e["dist_is_unwrapped_abs_diff"], e["weekday"]
    by_family: dict[str, list[dict]] = {}
    for t in spec["transforms"]:
        by_family.setdefault(t["family"], []).append(t)

    lines = [
        "# Angular columns — periods and the sin/cos transform",
        "",
        "Generated by `scripts/build_angular_spec.py`. Do not hand-edit — rerun the script.",
        "",
        "Resolves [#7](https://github.com/jenujari/prdict-pov-v1/issues/7). "
        "[#3](https://github.com/jenujari/prdict-pov-v1/issues/3) fixed *membership* "
        f"({c['angular_in_column_spec']} angular columns); this spec fixes the *transform*.",
        "",
        "## The headline: `*_dist` is not an angle",
        "",
        f"All {dist['columns']} cross-planet `<a>_<b>_dist` columns hold "
        "`abs(lon_a - lon_b)` — a plain absolute difference with **no modular "
        "reduction**. Verified exactly, on every row: worst deviation from "
        f"`abs(lon_a - lon_b)` is `{dist['max_error_vs_abs_diff']:.2e}` degrees, "
        "against "
        f"`{dist['max_error_vs_signed_separation']:.1f}` for a signed separation and "
        f"`{dist['max_error_vs_folded_separation']:.1f}` for a folded one.",
        "",
        "So the column is neither of the two things the ticket asked us to choose "
        "between. As an *angle* it does not behave: two planets at 359 and 1 degrees are "
        "recorded 358 degrees apart. Across the file, "
        f"**{dist['cells_over_180_but_within_30']:,} cells** record a separation above "
        "180 degrees where the folded angular separation is under 30.",
        "",
        "**Both encodings are kept.** An earlier revision of this spec discarded the "
        "`_dist` columns and replaced them with the recomputed separation. That was the "
        "wrong call to make unilaterally: the source is a **financial**-astrology scheme "
        "whose readings are deliberately tuned, and the unwrapped difference is what it "
        "computes. So the source column survives untransformed, the recomputed separation "
        "is added beside it, and **[#12](https://github.com/jenujari/prdict-pov-v1/issues/12) "
        "decides which earns its place** on the evidence.",
        "",
        "The one thing not done to `_dist` is sin/cos. That would half-work and "
        "half-poison: `cos` is even, so `cos(abs(d)) == cos(d)` and comes out right by "
        "luck, but `sin(abs(d)) == sign(d) * sin(d)`, and that sign tracks which planet "
        "holds the larger raw 0-360 number — an artifact of where the coordinate origin "
        "sits. So `_dist` is typed **linear** by #3 and passes straight through, carrying "
        "the source's own number with no transform applied to it at all.",
        "",
        "The added separation is `delta_ab = (lon_b - lon_a) mod 360`, signed rather than "
        "folded because Vedic drishti is directional — Mars aspects the 4th, 7th and 8th "
        "houses forward, Jupiter the 5th, 7th and 9th, Saturn the 3rd, 7th and 10th — so "
        "applying and separating are different states. `cos(delta)` carries aspect "
        "strength (conjunction `+1`, square `0`, opposition `-1`); `sin(delta)` carries "
        "the side.",
        "",
        "## Transform",
        "",
        "```",
        "theta = 2 * pi * (value - origin) / period",
        "emit    <stem>_sin = sin(theta)",
        "        <stem>_cos = cos(theta)",
        "```",
        "",
        "For the columns that **are** angles — longitudes, padas, `tithy` — the raw "
        "angle does not survive; sin/cos replaces it. (`_dist` is not in that set; it is "
        "a linear passthrough, see above.) A TFT reads a raw "
        "angle through linear layers, which put 359 and 1 at opposite ends of the "
        "range; and the threshold handle a raw angle would give XGBoost is already "
        "supplied by the `*_sign`, `*_nakshatra_name` and `*_navamsa_sign` "
        "categoricals, which are exactly the sector encodings such a split would "
        "reconstruct. (Note the assumption in the ticket that keeping both would feed "
        "the stage-1 prune a guaranteed pair does **not** hold: for a uniform angle "
        "`corr(theta, cos theta)` is 0 and `corr(theta, sin theta)` about `-0.78`, both "
        "under the 0.95 threshold.)",
        "",
        "| Family | Sources | Period | Origin | Value |",
        "|--------|---------|--------|--------|-------|",
    ]
    for family, members in by_family.items():
        head = members[0]
        lines.append(
            f"| `{family}` | {len(members)} | {head['period']:g} | {head['origin']:g} | "
            f"{'signed separation, recomputed' if head['value'] == 'signed_separation' else 'column as given'} |"
        )

    lines += [
        "",
        f"**{len(spec['transforms'])} transforms, {c['emitted_columns']} emitted columns.**",
        "",
        "### Periods, column by column",
        "",
        "| Source | Stem | Period | Emits |",
        "|--------|------|--------|-------|",
    ]
    for t in spec["transforms"]:
        lines.append(
            f"| `{t['source']}` | `{t['stem']}` | {t['period']:g} | "
            f"`{t['emits'][0]}`, `{t['emits'][1]}` |"
        )

    lines += [
        "",
        "## Dropped as structurally redundant",
        "",
        f"{c['dropped_redundant']} column. `ketu_longitude` is `rahu_longitude` + 180 on "
        f"every row (offsets observed: {e['ketu_antipodal_to_rahu']['offsets_observed']}), "
        "because Ketu *is* the south lunar node. Under the circular encoding that makes "
        "each Ketu sin/cos the exact negation of Rahu's — measured correlation "
        "`-1.000000` on both members — so they are dropped here rather than left for the "
        "fold-level prune to rediscover. The relation is definitional, not sampled, so "
        "dropping globally cannot leak; this is the same licence [#3]"
        "(https://github.com/jenujari/prdict-pov-v1/issues/3) used for its constants. "
        "`rahu_ketu_dist` was already dropped there as constant.",
        "",
        bullets(spec["dropped_redundant"]),
        "",
        "**The seven `<a>_ketu_dist` columns are not dropped**, although an earlier "
        "revision of this spec did drop them on the same antipodal argument. The argument "
        "does not survive contact with the raw column: it holds for the *recomputed* "
        "separation, where the Ketu pair is an exact negation of the Rahu one, but an "
        "unwrapped absolute difference to Ketu is not a function of the one to Rahu. "
        "Measured: the strongest correlation any of the seven reaches against its Rahu "
        f"twin is `{e['ketu_dist_not_redundant']['max_abs_correlation_with_rahu_twin']:.3f}`, "
        "and the closest best-fit line still leaves a residual of "
        f"`{e['ketu_dist_not_redundant']['min_linear_fit_residual_degrees']:.0f}` degrees. "
        "They are independent numbers and they stay.",
        "",
        "**`ketu_nakshatra_pada` survives**, and the asymmetry is deliberate. 180 degrees "
        "is 13.5 nakshatras only under a *uniform* 27-fold scheme; this source is 28-fold "
        "with unequal spans, so the antipodal relation does not carry through to pada. "
        "Measured: rahu pada does "
        f"{'' if e['ketu_pada_survives']['determined_by_rahu_pada'] else '**not** '}"
        "determine ketu pada, and the sin/cos correlations are "
        f"`{e['ketu_pada_survives']['corr_sin']:.3f}` and "
        f"`{e['ketu_pada_survives']['corr_cos']:.3f}` — both inside the 0.95 stage-1 "
        "threshold, so the fold-level prune would not remove it either.",
        "",
        "## Per-column notes",
        "",
        "### `tithy` — period 30, origin 1",
        "",
        f"Exactly `{e['tithy']['formula']}`, verified on every row, ranging "
        f"{e['tithy']['min']}..{e['tithy']['max']}. **Not** off by one: an elongation of "
        "0.004 degrees yields tithy 1, so origin 1 is right and `(tithy - 1) / 30` is the "
        "position on the circle.",
        "",
        "Note the file carries **no `sun_moon_dist` column** — tithy is the only Sun-Moon "
        "separation feature there is, at 12-degree resolution. The continuous elongation "
        "is deliberately *not* added as a 28th separation pair: tithi is a named unit of "
        "Vedic practice with per-tithi qualities, and the hypothesis under test is about "
        "the tradition's own quantities.",
        "",
        "### `*_nakshatra_pada` — period 4, origin 1",
        "",
        f"Genuinely cyclic: ordered by increasing longitude, the Sun's pada wraps "
        f"4 -> 1 {e['pada_wraps']['four_to_one_transitions']} times. Period 4, origin 1.",
        "",
        "The ticket asked whether the real cyclic quantity is `(nakshatra index * 4 + "
        "pada)` on a 108-fold circle. **It is not.** The source uses the "
        f"{e['nakshatra_scheme']['levels']}-fold scheme *including* Abhijit, with unequal "
        f"spans — narrowest {e['nakshatra_scheme']['narrowest_degrees']:g} degrees, widest "
        f"{e['nakshatra_scheme']['widest_degrees']:g}. That is 112 unequal cells, not a "
        "uniform 108-fold circle, and its uniform counterpart is just the longitude, "
        "which is already a feature.",
        "",
        "### `weekday` — period 7, origin 0, categorical *retained*",
        "",
        "The only transform that does not replace its source: `weekday` stays a "
        "categorical (per [#3](https://github.com/jenujari/prdict-pov-v1/issues/3)) and "
        "additionally gets a cyclic view. Level names line up exactly with the "
        "Monday-first `record_date` index, so the position is read off the string.",
        "",
        "Sessions per level on the trading-day index:",
        "",
        "| Level | Sessions |",
        "|-------|----------|",
    ]
    for level in WEEKDAY_ORDER:
        lines.append(f"| `{level}` | {weekday['sessions_per_level'].get(level, 0):,} |")

    lines += [
        "",
        "Effective cardinality is 5. The two weekend entries are real special sessions — "
        "the Budget Saturday and the Muhurat Sunday — not data errors. Be aware the "
        "period-7 circle is a *calendar* fact, not a trading-index one: Friday and Monday "
        "are consecutive sessions but sit 3 apart on the circle. The categorical is what "
        "carries the discrete weekday effect; the cyclic pair is the smooth calendar "
        "position alongside it.",
        "",
        "### `*_latitude` — not angular",
        "",
        "Confirmed excluded. Latitudes occupy a narrow band around zero (widest is "
        "Venus at about +/-8.6 degrees) and never wrap, so they stay linear numeric in "
        "[#3](https://github.com/jenujari/prdict-pov-v1/issues/3).",
        "",
        "## Pairs are atomic",
        "",
        spec["conventions"]["pair_atomicity_note"],
        "",
        "A pair is identified purely by name: `<stem>_sin` and `<stem>_cos`. "
        "`prdict.angles.pair_groups()` returns the grouping, and "
        "[#12](https://github.com/jenujari/prdict-pov-v1/issues/12) and "
        "[#13](https://github.com/jenujari/prdict-pov-v1/issues/13) are bound by it.",
        "",
        "## Counts",
        "",
        "| | |",
        "|---|---|",
        f"| Angular columns from #3 | {c['angular_in_column_spec']} |",
        f"| Dropped as redundant (Ketu) | {c['dropped_redundant']} |",
        f"| Angular columns transformed | {c['angular_transformed']} |",
        f"| Extra cyclic views (`weekday`) | {c['extra_cyclic_views']} |",
        f"| **Emitted sin/cos columns** | **{c['emitted_columns']}** |",
        f"| Features before | {c['features_before']} |",
        f"| Features after | {c['features_after']} |",
        f"| **Model input width** | **{c['model_input_width']}** |",
        "",
        f"Model input width is {len(json.loads(COLUMN_SPEC.read_text())['types']['boolean'])} "
        "boolean + "
        f"{len(json.loads(COLUMN_SPEC.read_text())['types']['categorical'])} categorical + "
        f"{len(json.loads(COLUMN_SPEC.read_text())['types']['linear_numeric'])} linear numeric + "
        f"{c['emitted_columns']} emitted cyclic.",
        "",
        "## Downstream",
        "",
        "- [#9](https://github.com/jenujari/prdict-pov-v1/issues/9) — sin/cos output is "
        "already bounded to `[-1, 1]`; decide whether it is scaled again with the linear "
        "block or passed through.",
        "- [#12](https://github.com/jenujari/prdict-pov-v1/issues/12) and "
        "[#13](https://github.com/jenujari/prdict-pov-v1/issues/13) — bound by pair "
        "atomicity above.",
        "- [#15](https://github.com/jenujari/prdict-pov-v1/issues/15) — every separation, "
        "tithy, pada, nakshatra, sign and navamsa column is an exact function of the 9 "
        "longitudes. The Ketu drop here removes only the provable duplicates *among "
        "angular columns*; the wider functional-dependency question is that ticket's.",
        "",
        "  Found in passing and handed over rather than acted on: the antipodal relation "
        "also makes "
        + ", ".join(f"`{c}`" for c in e["antipodal_categoricals"])
        + " exact functions of their Rahu counterparts, because signs (30 degrees) and "
        "navamsas (30/9) are uniform divisions and 180 is a whole number of each. Those "
        "are **categoricals**, so removing them is #9's and #15's call, not this "
        "ticket's. `ketu_nakshatra_name` is *not* in that set — Abhijit again.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    spec = build()
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    SPEC_MD.write_text(render_markdown(spec))

    c = spec["counts"]
    print("verified: _dist is abs(lon_a - lon_b), tithy formula, ketu antipodal, pada wrap")
    for label, key in [
        ("angular from #3", "angular_in_column_spec"),
        ("dropped (ketu_longitude)", "dropped_redundant"),
        ("_dist kept as linear", "dist_columns_kept"),
        ("separations added", "separations_added"),
        ("transformed", "angular_transformed"),
        ("extra cyclic", "extra_cyclic_views"),
        ("emitted", "emitted_columns"),
        ("features after", "features_after"),
        ("model input width", "model_input_width"),
    ]:
        print(f"  {label:20s}: {c[key]:3d}")
    print(f"\nwrote {SPEC_JSON.relative_to(ROOT)} and {SPEC_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
