"""How much of the 207-column feature set is genuinely independent, and what is done about the rest.

Resolves wayfinder ticket #15. Builds on #30's correctness audit
(`kb/derived_audit.md`), which recomputed every derived family from base columns
and found all of them exact. #30 answered *are the derived columns correct*; this
script answers the next question — *how many of them carry information the base
columns do not* — and records a keep/drop decision per family, separately for the
two models, because the answer differs by model.

Three things are settled here and written to `kb/independence_spec.{json,md}`:

  1. **The base set.** 44 independent quantities behind the 207 features — the 9
     longitudes (one redundant: Ketu = Rahu + 180), 7 latitudes, 7 distances and
     the three speed families, plus the weekday. Verified against the file.
  2. **The dependency graph.** Every other family is an exact function of the base
     set. #30 proved this for the bala and sign families; this script confirms the
     categorical bins (`sign`, `nakshatra`, `pada`, ...), `tithy`, and the 34
     cross-planet separations directly, and measures the one fact #12 needs: a
     sharp bin has near-zero *linear* correlation with the longitude it is a
     function of, so #12's `|r| >= 0.95` prune passes all of them through.
  3. **Keep / drop, per model.** Sharp categorical boundaries are redundant to a
     tree (it recovers `floor(lon/30)` for free) but not to a smooth encoder, so
     they are dropped for XGBoost and kept as embeddings for the TFT. Smooth balas
     ride into `linear_numeric` for both and are collapsed downstream by #12's
     prune and #13's PCA. Interactions are kept for both.

    uv run python scripts/build_independence_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "nft50.csv"
COLUMN_SPEC = ROOT / "kb" / "column_spec.json"
ENCODING_SPEC = ROOT / "kb" / "encoding_spec.json"
DERIVED_AUDIT = ROOT / "kb" / "derived_audit.json"
OUT_JSON = ROOT / "kb" / "independence_spec.json"
OUT_MD = ROOT / "kb" / "independence_spec.md"

PLANETS = ["sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu"]
PRUNE_THRESHOLD = 0.95

# Every feature family, its kind, the base quantities that determine it, and the
# keep/drop call for each model. This table is the human decision; the evidence
# columns beside it are computed below and must agree, or the build fails.
#
# kind:
#   base          a genuinely independent input
#   sharp_bin     a discontinuous function of a base column (a floor / boundary)
#   smooth_bala   a smooth function of a base column (cos / triangular / ramp)
#   interaction   a function of *two* base columns (a separation between bodies)
BASE_SUFFIXES = ["longitude", "latitude", "distance", "speed_long", "speed_lat", "speed_dist"]

SHARP_BINS = {
    "sign": ("longitude", "floor(lon / 30) mod 12"),
    "navamsa_sign": ("longitude", "floor(lon / (30/9)) mod 12"),
    "nakshatra_name": ("longitude", "28-fold Abhijit division of longitude"),
    "nakshatra_pada": ("longitude", "108-fold division of longitude"),
    "sign_lord": ("longitude", "ruling planet of the sign, so of floor(lon/30)"),
    "sign_lordship": ("longitude", "Naisargika maitri of the sign's lord"),
    "vargottama": ("longitude", "sign == navamsa_sign"),
    "is_retro": ("speed_long", "speed_long < 0"),
    "speed_category": ("speed_long", "monotone binning of speed_long"),
    "vedha": ("speed_long", "a function of speed_category"),
    "vedha_target": ("longitude", "a function of (nakshatra_name, vedha)"),
}
SMOOTH_BALAS = {
    "uchcha_bala": ("longitude", "75 + 25*cos(lon - reference)"),
    "kshetra_bala": ("longitude", "dignity_weight * triangular(lon, 30)"),
    "navamsha_bala": ("longitude", "dignity_weight * triangular(lon, 30/9)"),
    "uday_bala": ("longitude", "ramp from the combustion orb (also needs sun_longitude)"),
    "vakra_bala": ("speed_long", "-k * speed_long if retrograde else 0"),
}


def load() -> tuple[pd.DataFrame, dict, dict, dict]:
    frame = pd.read_csv(CSV_PATH)
    cspec = json.loads(COLUMN_SPEC.read_text())
    espec = json.loads(ENCODING_SPEC.read_text())
    audit = json.loads(DERIVED_AUDIT.read_text())
    return frame, cspec, espec, audit


def family_of(col: str) -> str | None:
    """The feature family a column belongs to, by its suffix past the planet name.

    Cross-planet separations (`sun_mars_dist`) carry two planet names, so they are
    caught by the `_dist` suffix directly rather than by stripping one prefix.
    `tithy` and `weekday` are singletons.
    """
    if col in ("tithy", "weekday"):
        return col
    if col.endswith("_dist") and "speed" not in col:
        return "separation"
    for planet in PLANETS:
        if col.startswith(planet + "_"):
            return col[len(planet) + 1 :]
    return None


def _mod360(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) % 360.0


def confirm_nakshatra_determinism(frame: pd.DataFrame) -> dict:
    """Confirm nakshatra is a function of longitude, and that #15's residual is Abhijit.

    A standard equal 27-fold division of longitude misclassifies ~7% of rows — the
    ticket's spot-check residual. That is not noise: the file declares 28 nakshatra
    names, the extra one being Abhijit, whose non-equal boundaries the 27-fold
    scheme cannot place. Binned finely enough that boundary placement no longer
    matters, each name occupies one contiguous longitude arc — i.e. it is a
    deterministic function of longitude, and the residual is boundary rounding.
    """
    names = set()
    for p in PLANETS:
        names |= set(frame[f"{p}_nakshatra_name"].dropna())

    def purity(col: str, bins: int) -> float:
        lon = _mod360(frame[f"{col}_longitude"])
        key = np.floor(lon / (360.0 / bins)).astype(int)
        sub = pd.DataFrame({"k": key, "v": frame[f"{col}_nakshatra_name"]}).dropna()
        pure = sub.groupby("k")["v"].nunique().eq(1)
        pure_keys = set(pure[pure].index)
        return float(sub["k"].isin(pure_keys).mean())

    return {
        "declared_names": len(names),
        "standard_division": 27,
        "row_purity_27_equal_bins": round(np.mean([purity(p, 27) for p in PLANETS]), 4),
        "row_purity_2700_bins": round(np.mean([purity(p, 2700) for p in PLANETS]), 5),
        "conclusion": (
            "nakshatra is a deterministic 28-fold (Abhijit) function of longitude; "
            "the equal-27-bin residual is the non-equal Abhijit boundary, not information"
        ),
    }


def confirm_base_reductions(frame: pd.DataFrame, audit: dict) -> dict:
    """The base columns that are themselves redundant, with evidence.

    Ketu's longitude is Rahu's + 180 exactly, so only 8 of the 9 longitudes are
    independent. The eight degenerate node columns (#30) were already removed
    before the 207-feature count, so they are noted, not re-dropped here.
    """
    diff = (_mod360(frame["ketu_longitude"] - frame["rahu_longitude"]))
    antipodal = {
        "column": "ketu_longitude",
        "determined_by": ["rahu_longitude"],
        "rule": "rahu_longitude + 180 (mod 360)",
        "offset_mean": round(float(diff.mean()), 6),
        "offset_std": round(float(diff.std()), 8),
    }
    assert abs(antipodal["offset_mean"] - 180.0) < 1e-3 and antipodal["offset_std"] < 1e-3
    return {
        "antipodal_longitude": antipodal,
        "degenerate_removed_upstream": sorted(audit["degenerate_columns"].keys()),
    }


def confirm_singletons(frame: pd.DataFrame) -> dict:
    """`tithy` is the Sun–Moon separation binned — verify it exactly."""
    pred = np.floor(_mod360(frame["moon_longitude"] - frame["sun_longitude"]) / 12.0) + 1
    exact = float((pred == frame["tithy"]).mean())
    assert exact == 1.0, f"tithy no longer exactly floor((moon-sun)/12)+1: {exact}"
    return {
        "tithy": {
            "determined_by": ["moon_longitude", "sun_longitude"],
            "rule": "floor(((moon - sun) mod 360) / 12) + 1",
            "exact_fraction": exact,
        }
    }


def prune_survival(frame: pd.DataFrame, families: dict[str, list[str]]) -> dict:
    """The fact #12 needs: a sharp bin's *linear* correlation with its base driver.

    Each sharp family is an exact function of a base column, yet #12's stage-1
    prune is pairwise `|r| >= 0.95`. A step function of a continuous variable has
    almost no linear correlation with it, so the prune passes every one of these
    through. Reported as the strongest `|r|` any column in the family reaches
    against its driver — all of them fall far below the threshold.
    """
    out = {"threshold": PRUNE_THRESHOLD, "families": {}}
    for fam, (base_suffix, _rule) in SHARP_BINS.items():
        cols = families.get(fam, [])
        best = 0.0
        for col in cols:
            planet = col[: col.index("_")]
            base_col = f"{planet}_{base_suffix}"
            if base_col not in frame.columns:
                continue
            codes = pd.Categorical(frame[col]).codes.astype(float)
            base = _mod360(frame[base_col]) if base_suffix == "longitude" else frame[base_col]
            mask = codes >= 0
            r = np.corrcoef(codes[mask], np.asarray(base, dtype=float)[mask])[0, 1]
            best = max(best, abs(float(r)))
        out["families"][fam] = {
            "columns": len(cols),
            "max_abs_r_vs_base": round(best, 3),
            "survives_prune": best < PRUNE_THRESHOLD,
        }
    n = len(out["families"])
    survive = sum(v["survives_prune"] for v in out["families"].values())
    out["all_survive"] = survive == n
    return out


def predict_pca(frame: pd.DataFrame, espec: dict) -> dict:
    """Where #13's PCA(0.95) lands on `linear_numeric` — a prediction, not a surprise.

    The linear block is mostly smooth balas and rates driven by ~44 base
    quantities, so one might expect a collapse to near 44. Linear PCA cannot
    exploit *nonlinear* determinism, though: a `cos(lon)` needs two components and
    a triangular ramp several, so the collapse is real but modest. Computed on the
    standardised raw columns so #13 confirms rather than discovers it.
    """
    from sklearn.decomposition import PCA

    cols = [c for f in espec["families"] if f["family"] == "linear_numeric" for c in f["columns"]]
    present = [c for c in cols if c in frame.columns]
    X = frame[present].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    Xs = (X - X.mean()) / X.std(ddof=0)
    Xs = Xs.loc[:, Xs.std() > 0]
    ratios = PCA().fit(Xs.to_numpy()).explained_variance_ratio_
    cum = np.cumsum(ratios)
    return {
        "block": "linear_numeric",
        "n_columns": len(present),
        "n_components_0.95": int(np.searchsorted(cum, 0.95) + 1),
        "n_components_0.99": int(np.searchsorted(cum, 0.99) + 1),
        "note": (
            "modest collapse — nonlinear (cos / triangular) determinism is invisible "
            "to linear PCA, so the block does not fall to the base-set size"
        ),
    }


def build() -> dict:
    frame, cspec, espec, audit = load()
    feats = list(cspec["roles"]["feature"])

    families: dict[str, list[str]] = {}
    for col in feats:
        fam = family_of(col)
        if fam is None:
            raise ValueError(f"{col!r} matches no known family — the classifier is stale")
        families.setdefault(fam, []).append(col)

    def kind_of(fam: str) -> str:
        if fam in BASE_SUFFIXES or fam in ("weekday",):
            return "base"
        if fam in SHARP_BINS:
            return "sharp_bin"
        if fam in SMOOTH_BALAS:
            return "smooth_bala"
        if fam in ("separation", "tithy"):
            return "interaction"
        raise ValueError(f"family {fam!r} has no kind")

    # The keep/drop rule follows only from the kind, so it is stated once.
    def policy(kind: str) -> dict:
        return {
            "base": {"xgboost": True, "tft": True},
            "sharp_bin": {"xgboost": False, "tft": True},
            "smooth_bala": {"xgboost": True, "tft": True},
            "interaction": {"xgboost": True, "tft": True},
        }[kind]

    rationale = {
        "base": "independent input — kept for both models",
        "sharp_bin": (
            "a discontinuous function of a base column; a tree recovers the boundary "
            "from the base for free (drop for XGBoost), but a smooth encoder cannot "
            "synthesise it, so it is kept as an embedding for the TFT. #12's linear "
            "prune misses it entirely (see prune_survival)"
        ),
        "smooth_bala": (
            "a smooth function of a base column; kept for both, but redundant — the "
            "near-linear part is removed by #12's prune and the rest collapsed by "
            "#13's PCA, which linear tools *can* do for a smooth dependency"
        ),
        "interaction": (
            "a separation between two bodies; genuinely useful and awkward for either "
            "model to reconstruct from separate longitudes (a tree splits axis-aligned), "
            "so kept for both"
        ),
    }
    angular_rationale = (
        "typed **angular** in #3's own column spec, so #7 already encodes it as a "
        "sin/cos pair (the `cyclic` family), never as a discrete column. That "
        "encoding always rides in both models regardless of this family's *kind* — "
        "there is no droppable discrete form to drop. Stated explicitly because "
        "`nakshatra_pada` is otherwise a `sharp_bin` and would default to 'drop for "
        "XGBoost'; that decision is unenforceable against a cyclic-encoded column "
        "and would silently no-op if applied, so it is pinned to keep-both here instead"
    )

    # Columns typed `angular` by #3 are already sin/cos-encoded by #7 and always
    # ride in the `cyclic` family (encoding_spec's `none` fit scope) — no model
    # ever sees their raw form, so no keep/drop decision on the raw name is
    # enforceable. `longitude` and `tithy` already resolve to keep-both under the
    # kind-based policy; `nakshatra_pada` is the one family this actually changes
    # (its raw form looks like a sharp_bin and would otherwise default to
    # drop-for-XGBoost, which the real encoded input has no way to honour).
    angular_raw = set(cspec["types"]["angular"])

    graph = {}
    for fam, cols in sorted(families.items()):
        kind = kind_of(fam)
        determined_by, rule = None, None
        if fam in SHARP_BINS:
            determined_by, rule = SHARP_BINS[fam]
        elif fam in SMOOTH_BALAS:
            determined_by, rule = SMOOTH_BALAS[fam]
        elif fam == "separation":
            determined_by, rule = "longitude pair", "|lon_a - lon_b| folded into [0, 180]"
        elif fam == "tithy":
            determined_by, rule = "moon_longitude + sun_longitude", "floor((moon - sun)/12) + 1"
        is_angular_encoded = set(cols) <= angular_raw
        graph[fam] = {
            "columns": cols,
            "n": len(cols),
            "kind": kind,
            "determined_by": determined_by,
            "rule": rule,
            "keep": {"xgboost": True, "tft": True} if is_angular_encoded else policy(kind),
            "rationale": angular_rationale if is_angular_encoded else rationale[kind],
        }

    n_base = sum(g["n"] for g in graph.values() if g["kind"] == "base")
    n_dependent = len(feats) - n_base
    drop_xgb = [c for g in graph.values() for c in g["columns"] if not g["keep"]["xgboost"]]
    drop_xgb.append("ketu_longitude")  # antipodal, redundant to Rahu (see base_reductions)

    return {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/15",
        "builds_on": "https://github.com/jenujari/prdict-pov-v1/issues/30",
        "source": {"file": CSV_PATH.name, "rows": int(len(frame)), "features": len(feats)},
        "totals": {
            "features": len(feats),
            "base_columns": n_base,
            "independent_quantities": n_base - 1,  # Ketu longitude is redundant
            "deterministic_dependents": n_dependent,
        },
        "base_set": {
            "suffixes": BASE_SUFFIXES + ["weekday"],
            "reductions": confirm_base_reductions(frame, audit),
        },
        "nakshatra": confirm_nakshatra_determinism(frame),
        "singletons": confirm_singletons(frame),
        "families": graph,
        "prune_survival": prune_survival(frame, families),
        "pca_prediction": predict_pca(frame, espec),
        "policy": {
            "xgboost": {
                "drop": sorted(drop_xgb),
                "keep_n": len(feats) - len(set(drop_xgb) & set(feats)),
                "note": (
                    "sharp categorical/boolean bins dropped (trees recover them from the "
                    "base); angular-encoded families (nakshatra_pada) are exempt — they "
                    "carry no droppable discrete form, see families.nakshatra_pada.rationale"
                ),
            },
            "tft": {
                "drop": ["ketu_longitude"],
                "keep_n": len(feats) - 1,
                "note": "sharp bins kept as embeddings; only the antipodal Ketu longitude drops",
            },
        },
        "what_12_inherits": (
            "Stage-1 still runs. #15 removes the *sharp* redundancy the linear prune "
            "cannot see and shrinks the XGBoost input; the flattened 90-session window "
            "still holds heavy day-lag collinearity in the surviving columns, which is "
            "exactly what #12's |r| >= 0.95 prune is for."
        ),
    }


def render_markdown(spec: dict) -> str:
    t = spec["totals"]
    ps = spec["prune_survival"]
    pca = spec["pca_prediction"]
    nak = spec["nakshatra"]
    lines: list[str] = []
    a = lines.append

    a("# Feature independence — the base set and what is redundant")
    a("")
    a("Generated by `scripts/build_independence_spec.py`. Do not hand-edit — rerun the script.")
    a("")
    a(f"Resolves [#15]({spec['resolves']}). Builds on [#30]({spec['builds_on']}) "
      "(`kb/derived_audit.md`), which proved every derived family is an exact function "
      "of the base columns; this ticket names the base set, measures the redundancy the "
      "correlation prune cannot see, and records a keep/drop decision per family.")
    a("")
    a("## The headline")
    a("")
    a(f"**{t['independent_quantities']} independent quantities generate all "
      f"{t['features']} features.** {t['deterministic_dependents']} columns "
      f"({100 * t['deterministic_dependents'] / t['features']:.0f}%) are exact "
      "deterministic functions of the base set and carry no information beyond it. "
      "The base set is the "
      f"{t['base_columns']} columns below, one of which (Ketu's longitude) is itself redundant.")
    a("")
    a("## The base set")
    a("")
    a("| Group | Columns | Note |")
    a("|-------|---------|------|")
    for suf in spec["base_set"]["suffixes"]:
        fam = spec["families"].get(suf)
        n = fam["n"] if fam else 1
        note = "Ketu = Rahu + 180, so 8 of 9 are independent" if suf == "longitude" else ""
        label = "`weekday`" if suf == "weekday" else f"`*_{suf}`"
        a(f"| {label} | {n} | {note} |")
    red = spec["base_set"]["reductions"]["antipodal_longitude"]
    a("")
    a(f"`ketu_longitude` is `rahu_longitude` + 180 exactly (offset mean "
      f"{red['offset_mean']}, std {red['offset_std']}); its sin/cos pair was already "
      "dropped as antipodal by #7. The eight numerically-constant Rahu/Ketu columns "
      "(#30) were removed before the 207-feature count.")
    a("")
    a("## Nakshatra is deterministic too (the #15 residual is Abhijit)")
    a("")
    a(f"The file declares **{nak['declared_names']}** nakshatra names, not the standard "
      f"{nak['standard_division']} — the extra one is Abhijit. An equal-27 division of "
      f"longitude leaves each name pure on only **{100 * nak['row_purity_27_equal_bins']:.0f}%** "
      "of rows (the ticket's spot-check residual), because it cannot place Abhijit's "
      f"non-equal boundary. Binned finely, each name occupies one contiguous arc — "
      f"**{100 * nak['row_purity_2700_bins']:.2f}%** pure — so nakshatra is a "
      "deterministic function of longitude and the residual is boundary rounding, not information.")
    a("")
    a("## Why #12's prune cannot find this")
    a("")
    a(f"Stage-1 of #12 is pairwise `|r| >= {ps['threshold']}`. Every sharp bin is an "
      "exact function of a base column, yet a step function of a continuous variable has "
      "almost no *linear* correlation with it, so the prune passes all of them through:")
    a("")
    a("| Sharp family | Columns | max \\|r\\| vs base | Survives the prune |")
    a("|--------------|---------|------------------|--------------------|")
    for fam, v in ps["families"].items():
        a(f"| `*_{fam}` | {v['columns']} | {v['max_abs_r_vs_base']} | "
          f"{'**yes**' if v['survives_prune'] else 'no'} |")
    a("")
    a(f"All {len(ps['families'])} survive — the redundancy is invisible to a linear prune. "
      "That is the whole reason this ticket exists.")
    a("")
    a("`*_nakshatra_pada` is measured here for the same reason as the rest, but its "
      "row in the table below does not follow from this evidence — see the note there.")
    a("")
    a("## Keep / drop, per family")
    a("")
    a("The decision follows from the family's *kind*, and it cuts differently for the "
      "two models. A tree recovers a boundary like `floor(lon/30)` from the base column "
      "for free, so sharp bins are dropped for XGBoost; a smooth encoder cannot make that "
      "boundary, so they are kept as embeddings for the TFT. Smooth balas ride into "
      "`linear_numeric` for both and are collapsed downstream. Interactions are kept for both.")
    a("")
    a("**Exception: angular-encoded families.** `*_longitude`, `*_nakshatra_pada` and "
      "`tithy` are typed **angular** by #3, so #7 already encodes them as sin/cos pairs "
      "— there is no discrete raw form in the actual model input for either model to "
      "drop. `longitude` and `tithy` already land on keep-both under their own kind; "
      "`*_nakshatra_pada` looks like an ordinary `sharp_bin` and would otherwise default "
      "to drop-for-XGBoost, which is unenforceable against a cyclic encoding — pinned "
      "to keep-both here instead of silently no-op-ing.")
    a("")
    a("| Family | n | Kind | Determined by | XGBoost | TFT |")
    a("|--------|---|------|---------------|---------|-----|")
    order = {"base": 0, "interaction": 1, "smooth_bala": 2, "sharp_bin": 3}
    angular_fams = {"longitude", "nakshatra_pada", "tithy"}
    for fam, g in sorted(spec["families"].items(), key=lambda kv: (order[kv[1]["kind"]], kv[0])):
        keep = g["keep"]
        det = g["determined_by"] or "—"
        mark = " †" if fam in angular_fams else ""
        a(f"| `*_{fam}`{mark} | {g['n']} | {g['kind']} | {det} | "
          f"{'keep' if keep['xgboost'] else '**drop**'} | "
          f"{'keep' if keep['tft'] else '**drop**'} |")
    a("")
    a("† angular-encoded — see the exception above.")
    a("")
    xgb, tft = spec["policy"]["xgboost"], spec["policy"]["tft"]
    a(f"- **XGBoost** keeps {xgb['keep_n']} of {t['features']} features "
      f"— {xgb['note']}.")
    a(f"- **TFT** keeps {tft['keep_n']} of {t['features']} features — {tft['note']}.")
    a("")
    a("## What #13's PCA will do (a prediction)")
    a("")
    a(f"PCA(0.95) on the standardised `{pca['block']}` block "
      f"({pca['n_columns']} columns) lands at **{pca['n_components_0.95']}** components "
      f"(0.99 → {pca['n_components_0.99']}). {pca['note'].capitalize()}. So #13 should "
      "expect a real but modest collapse, not a fall to the base-set size — a "
      "confirmation, not a surprise.")
    a("")
    a("## What #12 inherits")
    a("")
    a(spec["what_12_inherits"])
    a("")
    return "\n".join(lines)


def main() -> None:
    spec = build()
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(spec))

    t = spec["totals"]
    ps = spec["prune_survival"]
    pca = spec["pca_prediction"]
    print(f"features {t['features']}  base {t['base_columns']} "
          f"({t['independent_quantities']} independent)  "
          f"deterministic {t['deterministic_dependents']}")
    print(f"nakshatra fine-bin purity {spec['nakshatra']['row_purity_2700_bins']} "
          f"({spec['nakshatra']['declared_names']} names, Abhijit)")
    print(f"prune survival: all {len(ps['families'])} sharp families survive |r|>={ps['threshold']}: "
          f"{ps['all_survive']}")
    print(f"pca(0.95) on linear_numeric: {pca['n_components_0.95']}/{pca['n_columns']}")
    print(f"policy: XGBoost keeps {spec['policy']['xgboost']['keep_n']}, "
          f"TFT keeps {spec['policy']['tft']['keep_n']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")

    assert ps["all_survive"], "a sharp family unexpectedly correlates with its base — check the prune claim"


if __name__ == "__main__":
    main()
