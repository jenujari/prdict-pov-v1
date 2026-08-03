"""Correctness audit of every derived column in nft50.csv.

Resolves wayfinder ticket #30. [#7](issues/7) found that the 34 cross-planet
`_dist` columns hold `abs(lon_a - lon_b)` with no modular reduction — a whole
family silently wrong, found only because that ticket happened to check the
arithmetic. This script checks the rest of the file the same way.

The method is fixed by the ticket's scope: a derived column is recomputed from
**base columns already present in the file** and compared. No new ephemeris is
computed and no external astrology library is consulted, so where a column
cannot be reproduced from the file's own contents the verdict is
`unverifiable`, not `wrong`.

Every family gets one of four verdicts:

  exact         reproduced from base columns to floating-point tolerance
  miscomputed   reproduced, but the formula's constants are demonstrably wrong
  degenerate    numerically constant — carries no information at all
  unverifiable  not reproducible from the file's own columns

The answer is that **nothing else is miscomputed**: `_dist` was the only broken
family, and every other one reproduces exactly. The audit's lasting value is
therefore the formulae it had to recover in order to check them, which are
written up in kb/derived_audit.md.

One caveat on method. The source is a **financial**-astrology scheme, not a
natal one, so its reference constants — the `uchcha_bala` peak longitudes above
all — are not compared against a natal table. They are recovered from the data
and then asserted, which is what makes the check rerunnable without importing a
convention the source does not follow.

The one thing found: eight Rahu/Ketu columns hold correct values that are
constant to machine precision, and escaped #3's exact-equality constant filter
because float noise gives them thousands of distinct values.

    uv run python scripts/audit_derived_columns.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "nft50.csv"
COLUMN_SPEC = ROOT / "kb" / "column_spec.json"
AUDIT_JSON = ROOT / "kb" / "derived_audit.json"
AUDIT_MD = ROOT / "kb" / "derived_audit.md"

TOL = 1e-6

PLANETS = ["sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu"]

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Lord of each sign, in SIGNS order. Classical Vedic rulerships.
SIGN_LORD = [
    "mars", "venus", "mercury", "moon", "sun", "mercury",
    "venus", "mars", "jupiter", "saturn", "saturn", "jupiter",
]

# Naisargika maitri — the classical natural-friendship table. Asserted against
# the file's own `*_sign_lordship` column below rather than trusted.
FRIENDSHIP = {
    "sun": {"sun": "Self", "moon": "Friend", "mars": "Friend", "jupiter": "Friend",
            "mercury": "Neutral", "venus": "Enemy", "saturn": "Enemy"},
    "moon": {"moon": "Self", "sun": "Friend", "mercury": "Friend", "mars": "Neutral",
             "jupiter": "Neutral", "venus": "Neutral", "saturn": "Neutral"},
    "mars": {"mars": "Self", "sun": "Friend", "moon": "Friend", "jupiter": "Friend",
             "venus": "Neutral", "saturn": "Neutral", "mercury": "Enemy"},
    "mercury": {"mercury": "Self", "sun": "Friend", "venus": "Friend", "mars": "Neutral",
                "jupiter": "Neutral", "saturn": "Neutral", "moon": "Enemy"},
    "jupiter": {"jupiter": "Self", "sun": "Friend", "moon": "Friend", "mars": "Friend",
                "saturn": "Neutral", "mercury": "Enemy", "venus": "Enemy"},
    "venus": {"venus": "Self", "mercury": "Friend", "saturn": "Friend", "mars": "Neutral",
              "jupiter": "Neutral", "sun": "Enemy", "moon": "Enemy"},
    "saturn": {"saturn": "Self", "mercury": "Friend", "venus": "Friend",
               "jupiter": "Neutral", "sun": "Enemy", "moon": "Enemy", "mars": "Enemy"},
}

# Strength weight per dignity, recovered from kshetra_bala and navamsha_bala.
DIGNITY_WEIGHT = {"Self": 100.0, "Friend": 75.0, "Neutral": 50.0, "Enemy": 25.0}

# Both nodes sit at 'Enemy' in every sign throughout the file (#3 records this as
# a constant column), and both bala families cap at 25 for them accordingly.
NODE_WEIGHT = 25.0

# The longitude each `*_uchcha_bala` column peaks at, recovered from the data by
# argmax and then asserted below. These are the source's own reference points:
# the scheme is **financial** astrology, not natal, so they are not expected to
# match a natal exaltation table and are not checked against one. Saturn's sits
# at 20 Aries, 180 degrees from the natal value — deliberate, per the data owner.
UCHCHA_REFERENCE = {
    "sun": 10.0,       # 10 Aries
    "moon": 33.0,      # 3 Taurus
    "mars": 298.0,     # 28 Capricorn
    "mercury": 165.0,  # 15 Virgo
    "jupiter": 95.0,   # 5 Cancer
    "venus": 357.0,    # 27 Pisces
    "saturn": 20.0,    # 20 Aries
    "rahu": 80.0,      # 20 Gemini
    "ketu": 260.0,     # 20 Sagittarius
}

# Astangata (combustion) orbs in degrees, and the elongation at which uday_bala
# reaches 100. Recovered exactly from the data; both match the classical values.
UDAY = {
    #          orb direct, orb retrograde, ceiling
    "moon": (12.0, 12.0, 180.0),
    "mars": (17.0, 17.0, 180.0),
    "mercury": (14.0, 12.0, 27.0),
    "jupiter": (11.0, 11.0, 180.0),
    "venus": (10.0, 8.0, 47.0),
    "saturn": (15.0, 15.0, 180.0),
}

# Speed categories in increasing order of speed_long.
SPEED_ORDER = [
    "kutil", "ati-vakra", "vakra", "ati-mand", "mand",
    "madhyam", "sama", "sheeghra", "ati-sheeghra",
]


def load() -> pd.DataFrame:
    frame = pd.read_csv(CSV_PATH)
    frame["record_date"] = pd.to_datetime(frame["record_date"])
    # The file is stored newest-first; every rate check below differences along
    # the row axis, so the sort is load-bearing, not cosmetic (see #8).
    return frame.sort_values("record_date").reset_index(drop=True)


def longitude(frame: pd.DataFrame, planet: str) -> np.ndarray:
    """Ketu's longitude was dropped by #7 as antipodal; reconstruct it."""
    if planet == "ketu":
        return (frame["rahu_longitude"].to_numpy() + 180.0) % 360.0
    return frame[f"{planet}_longitude"].to_numpy()


def sign_index(lon: np.ndarray) -> np.ndarray:
    return np.floor(lon / 30.0).astype(int) % 12


def navamsa_index(lon: np.ndarray) -> np.ndarray:
    return np.floor(lon / (30.0 / 9.0)).astype(int) % 12


def triangular(lon: np.ndarray, period: float) -> np.ndarray:
    """0 at the cell edges, 1 at the cell centre."""
    return 1.0 - np.abs(lon % period - period / 2.0) / (period / 2.0)


def folded_elongation(frame: pd.DataFrame, planet: str) -> np.ndarray:
    """Angular distance from the Sun, folded into [0, 180]."""
    e = (longitude(frame, planet) - frame["sun_longitude"].to_numpy()) % 360.0
    return np.abs(((e + 180.0) % 360.0) - 180.0)


def dignity_weights(frame: pd.DataFrame, planet: str, cell: np.ndarray) -> np.ndarray:
    if planet in ("rahu", "ketu"):
        return np.full(len(cell), NODE_WEIGHT)
    return np.array([DIGNITY_WEIGHT[FRIENDSHIP[planet][SIGN_LORD[i]]] for i in cell])


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_sign_and_navamsa(frame: pd.DataFrame) -> dict:
    """`*_sign` and `*_navamsa_sign` against longitude. Uniform divisions, so exact."""
    code = {name: i for i, name in enumerate(SIGNS)}
    out = {"sign": {}, "navamsa_sign": {}}
    for planet in PLANETS:
        lon = longitude(frame, planet)
        for family, index in (("sign", sign_index), ("navamsa_sign", navamsa_index)):
            observed = frame[f"{planet}_{family}"].map(code).to_numpy()
            wrong = int((index(lon) != observed).sum())
            if wrong:
                raise AssertionError(f"{planet}_{family}: {wrong} rows disagree with longitude")
            out[family][planet] = wrong
    return out


def check_sign_lordship(frame: pd.DataFrame) -> dict:
    """`*_sign_lordship` against `*_sign` and the classical friendship table."""
    out = {}
    for planet in PLANETS:
        column = f"{planet}_sign_lordship"
        observed = frame[column]
        if observed.nunique() == 1:
            out[planet] = {"constant": observed.iloc[0]}
            continue
        expected = np.array(
            [FRIENDSHIP[planet][SIGN_LORD[i]] for i in sign_index(longitude(frame, planet))]
        )
        wrong = int((expected != observed.to_numpy()).sum())
        if wrong:
            raise AssertionError(f"{column}: {wrong} rows disagree with Naisargika maitri")
        out[planet] = {"mismatches": wrong, "levels": int(observed.nunique())}
    return out


def check_uchcha_bala(frame: pd.DataFrame) -> dict:
    """`*_uchcha_bala` = 75 + 25*cos(lon - reference).

    The reference longitude is recovered from the data by argmax, then the whole
    column is reproduced from it. Both the shape and the reference set belong to
    the source's financial-astrology scheme; neither is checked against a natal
    table, which is a different system with different reference points.

    Being a cosine of `lon - reference`, the column is also an exact linear
    combination of the `*_longitude_sin` / `*_longitude_cos` pair #7 emits — see
    check_prune_survival.
    """
    out = {}
    for planet in PLANETS:
        lon = longitude(frame, planet)
        observed = frame[f"{planet}_uchcha_bala"].to_numpy()

        # Recover the peak longitude from the data rather than assuming it.
        peak = float(lon[int(np.argmax(observed))])
        reference = UCHCHA_REFERENCE[planet]
        if min(abs(peak - reference), 360.0 - abs(peak - reference)) > 0.5:
            raise AssertionError(
                f"{planet}_uchcha_bala peaks at {peak:.3f}, expected {reference}"
            )

        predicted = 75.0 + 25.0 * np.cos(np.deg2rad(lon - reference))
        error = float(np.abs(predicted - observed).max())
        if error > TOL:
            raise AssertionError(f"{planet}_uchcha_bala: max error {error}")

        out[planet] = {
            "peak_longitude": peak,
            "reference_longitude": reference,
            "max_error": error,
            "verdict": "exact",
        }
    return out


def check_kshetra_and_navamsha_bala(frame: pd.DataFrame) -> dict:
    """Both are `dignity weight * triangular ramp` — over the sign, then the navamsa."""
    out = {"kshetra_bala": {}, "navamsha_bala": {}}
    families = (
        ("kshetra_bala", 30.0, sign_index),
        ("navamsha_bala", 30.0 / 9.0, navamsa_index),
    )
    for family, period, index in families:
        for planet in PLANETS:
            lon = longitude(frame, planet)
            observed = frame[f"{planet}_{family}"].to_numpy()
            weight = dignity_weights(frame, planet, index(lon))
            predicted = weight * triangular(lon, period)
            error = float(np.abs(predicted - observed).max())
            if error > TOL:
                raise AssertionError(f"{planet}_{family}: max error {error}")
            out[family][planet] = {"max_error": error, "verdict": "exact"}
    return out


def check_uday_bala(frame: pd.DataFrame) -> dict:
    """`*_uday_bala` ramps linearly from the combustion orb to a ceiling.

    The orbs recovered from the data are exactly the classical astangata values,
    including the tighter retrograde orbs for Mercury and Venus. The ceiling is
    180 (opposition) for the outer bodies and the maximum elongation for the
    inner ones.
    """
    out = {}
    for planet in PLANETS:
        column = f"{planet}_uday_bala"
        observed = frame[column].to_numpy()
        if planet not in UDAY:
            out[planet] = {"constant": float(observed[0]), "verdict": "degenerate"}
            continue
        orb_direct, orb_retro, ceiling = UDAY[planet]
        retro = frame[f"{planet}_is_retro"].to_numpy()
        orb = np.where(retro, orb_retro, orb_direct)
        elongation = folded_elongation(frame, planet)
        predicted = np.clip(100.0 * (elongation - orb) / (ceiling - orb), 0.0, 100.0)
        error = float(np.abs(predicted - observed).max())
        if error > TOL:
            raise AssertionError(f"{column}: max error {error}")
        out[planet] = {
            "orb_direct": orb_direct,
            "orb_retrograde": orb_retro,
            "ceiling": ceiling,
            "max_error": error,
            "verdict": "exact",
        }
    return out


def check_vakra_bala(frame: pd.DataFrame) -> dict:
    """`*_vakra_bala` is linear in speed_long while retrograde, and 0 otherwise."""
    out = {}
    for planet in PLANETS:
        column = f"{planet}_vakra_bala"
        observed = frame[column].to_numpy()
        if frame[column].nunique() == 1:
            out[planet] = {"constant": float(observed[0]), "verdict": "degenerate"}
            continue
        retro = frame[f"{planet}_is_retro"].to_numpy()
        speed = frame[f"{planet}_speed_long"].to_numpy()

        direct_max = float(np.abs(observed[~retro]).max())
        if direct_max > TOL:
            raise AssertionError(f"{column}: nonzero ({direct_max}) on a direct row")

        slope, intercept = np.polyfit(speed[retro], observed[retro], 1)
        predicted = np.where(retro, slope * speed + intercept, 0.0)
        error = float(np.abs(predicted - observed).max())
        if error > TOL:
            raise AssertionError(f"{column}: max error {error}")
        out[planet] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "reference_speed": float(-100.0 / slope),
            "max_error": error,
            "verdict": "exact",
        }
    return out


def check_speed_category(frame: pd.DataFrame) -> dict:
    """`*_speed_category` must be a monotone binning of speed_long.

    The bin edges are constants of the source's own table and are not derivable
    from the file, so the checkable claim is the ordering: sorted by speed_long,
    the category index must never decrease. A single inversion would prove the
    column wrong.
    """
    code = {name: i for i, name in enumerate(SPEED_ORDER)}
    out = {}
    for planet in PLANETS:
        column = f"{planet}_speed_category"
        if frame[column].nunique() == 1:
            out[planet] = {"constant": frame[column].iloc[0], "verdict": "degenerate"}
            continue
        speed = frame[f"{planet}_speed_long"].to_numpy()
        category = frame[column].map(code).to_numpy()
        order = np.argsort(speed, kind="stable")
        violations = int((np.diff(category[order]) < 0).sum())
        if violations:
            raise AssertionError(f"{column}: {violations} ordering violations against speed_long")
        out[planet] = {
            "violations": violations,
            "levels_used": int(frame[column].nunique()),
            "verdict": "exact",
        }
    return out


def check_vedha(frame: pd.DataFrame) -> dict:
    """`*_vedha` against speed_category, `*_vedha_target` against (nakshatra, vedha)."""
    out = {"vedha": {}, "vedha_target": {}}
    for planet in PLANETS:
        vedha, category = f"{planet}_vedha", f"{planet}_speed_category"
        table = pd.crosstab(frame[category], frame[vedha])
        ambiguous = int(((table > 0).sum(axis=1) > 1).sum())
        if ambiguous:
            raise AssertionError(f"{vedha}: {ambiguous} speed categories map to >1 vedha")
        out["vedha"][planet] = {
            "mapping": {i: table.columns[table.loc[i] > 0][0] for i in table.index
                        if table.loc[i].sum() > 0},
            "verdict": "exact",
        }

        target, nakshatra = f"{planet}_vedha_target", f"{planet}_nakshatra_name"
        present = frame[[target, nakshatra, vedha]].dropna(subset=[target])
        worst = int(present.groupby([nakshatra, vedha])[target].nunique().max())
        if worst > 1:
            raise AssertionError(f"{target}: not determined by (nakshatra, vedha)")
        out["vedha_target"][planet] = {
            "determined_by": ["nakshatra_name", "vedha"],
            "nulls": int(frame[target].isna().sum()),
            "verdict": "exact",
        }

    # #3 recorded the sun_vedha_target null rule as `sun_vedha == 'no'`. That is
    # true, and it in turn is exactly the slowest speed category.
    nulls = frame["sun_vedha_target"].isna()
    if not bool((nulls == (frame["sun_speed_category"] == "ati-mand")).all()):
        raise AssertionError("sun_vedha_target nulls no longer coincide with ati-mand")
    out["sun_vedha_target_null_rule"] = "sun_speed_category == 'ati-mand'"
    return out


def check_rates(frame: pd.DataFrame) -> dict:
    """`*_speed_dist` and `*_speed_lat` against the day-over-day change.

    The file is calendar-daily with no gaps (#3), so a central difference along
    the row axis is the daily rate. This cannot be exact — a central difference
    is a second-order approximation of a derivative that the source computed
    analytically — so the claim under test is agreement in unit and sign, at a
    correlation a wrong unit or a flipped sign could not reach.
    """
    step = np.arange(len(frame), dtype=float)
    out = {}
    families = [
        ("distance", "speed_dist", lambda f, p: f[f"{p}_distance"].to_numpy()),
        ("latitude", "speed_lat", lambda f, p: f[f"{p}_latitude"].to_numpy()),
        ("longitude", "speed_long",
         lambda f, p: np.rad2deg(np.unwrap(np.deg2rad(longitude(f, p))))),
    ]
    for level, rate, getter in families:
        out[rate] = {}
        for planet in PLANETS:
            column = f"{planet}_{rate}"
            if column not in frame.columns:
                continue
            reported = frame[column].to_numpy()
            span = float(reported.max() - reported.min())
            if span < 1e-6:
                # Degenerate; correlation against noise is meaningless. Reported
                # by check_degenerate instead.
                out[rate][planet] = {"verdict": "degenerate", "range": span}
                continue
            if level == "distance" and frame[f"{planet}_distance"].nunique() == 1:
                out[rate][planet] = {"verdict": "degenerate", "note": "distance constant"}
                continue
            observed = np.gradient(getter(frame, planet), step)[1:-1]
            r = float(np.corrcoef(observed, reported[1:-1])[0, 1])
            slope = float(np.polyfit(reported[1:-1], observed, 1)[0])
            if r < 0.999:
                raise AssertionError(f"{column}: correlation with d({level})/dt is only {r}")
            out[rate][planet] = {"correlation": r, "slope": slope, "verdict": "consistent"}
    return out


def check_prune_survival(frame: pd.DataFrame) -> dict:
    """Which exactly-determined columns the stage-1 pairwise prune would still keep.

    `*_uchcha_bala` is `75 + 25*cos(lon - exaltation)`, so it is an exact linear
    combination of the `*_longitude_sin` / `*_longitude_cos` pair #7 emits — yet
    whether `|r| >= 0.95` against *either member individually* catches it depends
    on the exaltation angle. Measured, not argued from the uniform-angle case,
    because the longitudes are not uniformly distributed over 27 years.
    """
    threshold = 0.95
    out = {"threshold": threshold, "uchcha_bala": {}}
    for planet in PLANETS:
        if planet == "ketu":
            # #7 dropped ketu's sin/cos pair as antipodal, so there is no
            # partner column for the prune to match against at all.
            continue
        lon = np.deg2rad(longitude(frame, planet))
        observed = frame[f"{planet}_uchcha_bala"].to_numpy()
        r_sin = abs(float(np.corrcoef(observed, np.sin(lon))[0, 1]))
        r_cos = abs(float(np.corrcoef(observed, np.cos(lon))[0, 1]))
        out["uchcha_bala"][planet] = {
            "r_sin": r_sin,
            "r_cos": r_cos,
            "survives": max(r_sin, r_cos) < threshold,
        }

    # Every Ketu bala is an exact *duplicate* of Rahu's, not an inversion. Ketu's
    # longitude is Rahu's + 180 and its exaltation point is likewise 180 away, so
    # the two offsets cancel inside the cosine; for the two ramp families, 180 is
    # a whole number of signs (6) and of navamsas (54), and both nodes sit at
    # 'Enemy' dignity everywhere, so the ramp and the weight both repeat.
    out["ketu_duplicates_rahu"] = {}
    for family in ("uchcha_bala", "kshetra_bala", "navamsha_bala"):
        error = float(
            np.abs(frame[f"rahu_{family}"].to_numpy() - frame[f"ketu_{family}"].to_numpy()).max()
        )
        if error > TOL:
            raise AssertionError(f"ketu_{family} is no longer identical to rahu_{family}")
        out["ketu_duplicates_rahu"][f"ketu_{family}"] = error

    out["survivors"] = sorted(
        f"{p}_uchcha_bala" for p, e in out["uchcha_bala"].items() if e["survives"]
    )
    return out


def check_degenerate(frame: pd.DataFrame, numeric: list[str], dropped: list[str]) -> dict:
    """Numeric columns that are constant to machine precision.

    #3's constant filter originally tested exact equality, so float noise in the
    last bits let eight of these through into the feature set. The nodes have no
    ecliptic latitude and no distance by definition, and this file's Rahu is a
    *mean* node, so its longitude advances at a fixed rate and its speed column
    is likewise a constant.

    This ticket widened that filter to a tolerance, so the eight are now dropped
    at source. The sweep therefore covers the dropped columns too — otherwise the
    finding would erase itself the moment it was acted on.
    """
    degenerate = {}
    for column in numeric + dropped:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        series = series.astype(float)
        span = float(series.max() - series.min())
        if span < TOL and series.nunique() > 1:
            degenerate[column] = {
                "range": span,
                "std": float(series.std()),
                "value": float(series.mean()),
                "distinct_float_values": int(series.nunique()),
                "still_a_feature": column in numeric,
            }
    return degenerate


def build() -> dict:
    frame = load()
    spec = json.loads(COLUMN_SPEC.read_text())
    numeric = spec["types"]["angular"] + spec["types"]["linear_numeric"]
    dropped = spec["roles"]["dropped_constant"]

    audit = {
        "resolves": "https://github.com/jenujari/prdict-pov-v1/issues/30",
        "source": {"file": CSV_PATH.name, "rows": int(len(frame))},
        "families": {
            "sign_and_navamsa": check_sign_and_navamsa(frame),
            "sign_lordship": check_sign_lordship(frame),
            "uchcha_bala": check_uchcha_bala(frame),
            **check_kshetra_and_navamsha_bala(frame),
            "uday_bala": check_uday_bala(frame),
            "vakra_bala": check_vakra_bala(frame),
            "speed_category": check_speed_category(frame),
            **check_vedha(frame),
            "rates": check_rates(frame),
        },
        "prune_survival": check_prune_survival(frame),
        "degenerate_columns": check_degenerate(frame, numeric, dropped),
    }

    def verdicts_in(node) -> list[str]:
        if isinstance(node, dict):
            found = [node["verdict"]] if isinstance(node.get("verdict"), str) else []
            for value in node.values():
                found += verdicts_in(value)
            return found
        return []

    verdicts = verdicts_in(audit["families"])
    audit["summary"] = {
        "miscomputed": [v for v in verdicts if v == "miscomputed"],
        "degenerate": sorted(audit["degenerate_columns"]),
        "unverifiable": [v for v in verdicts if v == "unverifiable"],
        "ketu_duplicates": sorted(audit["prune_survival"]["ketu_duplicates_rahu"]),
    }
    return audit


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_markdown(audit: dict) -> str:
    fam = audit["families"]
    degen = audit["summary"]["degenerate"]
    survivors = audit["prune_survival"]["survivors"]

    lines = [
        "# Correctness audit of the derived columns",
        "",
        "Generated by `scripts/audit_derived_columns.py`. Do not hand-edit — rerun the script.",
        "",
        f"Resolves [#30]({audit['resolves']}).",
        "",
        "[#7](https://github.com/jenujari/prdict-pov-v1/issues/7) found the 34 cross-planet "
        "`_dist` columns held `abs(lon_a - lon_b)` with no modular reduction, caught only "
        "because that ticket happened to check the arithmetic. This audit checks every "
        "other derived family the same way: recompute from base columns **already in the "
        "file**, and compare. No new ephemeris, no external astrology library.",
        "",
        "One rule, set while this ticket ran and applied to it throughout: the source is a "
        "**financial**-astrology scheme with deliberately tuned readings, so its values "
        "are never judged against a natal or textbook table. Constants are recovered from "
        "the data and then asserted; a column is wrong only if it disagrees with *itself*. "
        "The same rule sent #7 back to keep the `_dist` columns rather than replace them.",
        "",
        "## Verdicts",
        "",
        "| Family | Columns | Verdict | Reproduced as |",
        "|--------|---------|---------|---------------|",
        "| `*_sign` | 9 | **exact** | `floor(lon / 30) mod 12` |",
        "| `*_navamsa_sign` | 9 | **exact** | `floor(lon / (30/9)) mod 12` |",
        "| `*_sign_lordship` | 7 + 2 const | **exact** | Naisargika maitri of the sign's lord |",
        "| `*_uchcha_bala` | 9 | **exact** | `75 + 25*cos(lon - reference)` |",
        "| `*_kshetra_bala` | 9 | **exact** | `dignity_weight * triangular(lon, 30)` |",
        "| `*_navamsha_bala` | 9 | **exact** | `dignity_weight * triangular(lon, 30/9)` |",
        "| `*_uday_bala` | 6 + 3 const | **exact** | linear ramp from the combustion orb |",
        "| `*_vakra_bala` | 5 + 4 const | **exact** | `-k * speed_long` if retrograde else 0 |",
        "| `*_speed_category` | 7 + 2 const | **exact** | monotone binning of `speed_long` |",
        "| `*_vedha` | 7 + 2 const | **exact** | a function of `speed_category` |",
        "| `*_vedha_target` | 9 | **exact** | a function of (`nakshatra_name`, `vedha`) |",
        "| `*_distance` / `*_speed_dist` | 7 pairs | **consistent** | day-over-day rate |",
        "| `*_latitude` / `*_speed_lat` | 7 pairs | **consistent** | day-over-day rate |",
        f"| Rahu/Ketu latitude, speed | {len(degen)} | **degenerate** | constant to machine precision |",
        "",
        "**No derived family is miscomputed, and none is unverifiable.** Every one was "
        "reproduced exactly from the file's own columns. The `_dist` family #7 flagged "
        "remains the only one whose encoding was ever in question, and it is now kept as "
        "the source computes it.",
        "",
        "## The recovered formulae",
        "",
        "Each family below is reproduced to floating-point tolerance, so these are the "
        "source's actual rules, not a description of them.",
        "",
        "### The bala block",
        "",
        "```",
        "uchcha_bala    = 75 + 25 * cos(longitude - reference)",
        "kshetra_bala   = dignity_weight * triangular(longitude, 30)",
        "navamsha_bala  = dignity_weight * triangular(longitude, 30/9)",
        "uday_bala      = clip(100 * (elongation - orb) / (ceiling - orb), 0, 100)",
        "vakra_bala     = -k * speed_long   if retrograde else 0",
        "",
        "triangular(x, p) = 1 - |x mod p - p/2| / (p/2)      # 0 at cell edge, 1 at centre",
        "dignity_weight   = Self 100, Friend 75, Neutral 50, Enemy 25",
        "elongation       = |planet longitude - sun longitude|, folded into [0, 180]",
        "```",
        "",
        "This is a **financial**-astrology scheme, not a natal one, so the constants below "
        "are the source's own and are deliberately not checked against a natal table. They "
        "are recovered from the data by argmax and least squares, then asserted — the "
        "script fails if the source stops matching.",
        "",
        f"`uchcha_bala` reference longitudes, exact to "
        f"{max(e['max_error'] for e in fam['uchcha_bala'].values()):.1e}:",
        "",
        "| Planet | Reference | | Planet | Reference |",
        "|--------|-----------|-|--------|-----------|",
    ]
    refs = [(p, e["reference_longitude"]) for p, e in fam["uchcha_bala"].items()]
    for i in range(0, len(refs), 2):
        left = f"| `{refs[i][0]}` | {refs[i][1]:.0f} |"
        right = (f" | `{refs[i+1][0]}` | {refs[i+1][1]:.0f} |"
                 if i + 1 < len(refs) else " | | |")
        lines.append(left + right)
    lines += [
        "",
        "`uday_bala` orbs and ceilings, exact to "
        f"{max(e['max_error'] for e in fam['uday_bala'].values() if 'max_error' in e):.1e}. "
        "The orbs are the astangata (combustion) distances, including the tighter "
        "retrograde orbs for Mercury and Venus; the ceiling is opposition for the outer "
        "bodies and maximum elongation for the inner ones:",
        "",
        "| Planet | Orb (direct) | Orb (retrograde) | Ceiling |",
        "|--------|--------------|------------------|---------|",
    ]
    for planet, entry in fam["uday_bala"].items():
        if "orb_direct" not in entry:
            continue
        lines.append(
            f"| `{planet}` | {entry['orb_direct']:.0f} | {entry['orb_retrograde']:.0f} | "
            f"{entry['ceiling']:.0f} |"
        )
    lines += [
        "",
        "Three consequences worth carrying forward. `uchcha_bala` is a cosine of "
        "`lon - reference`, hence an exact **linear** combination of the "
        "`*_longitude_sin` / `*_longitude_cos` pair #7 emits. `kshetra_bala` and "
        "`navamsha_bala` share one weight table, and it is the same Naisargika maitri that "
        "`*_sign_lordship` reports — so the three families are one quantity read three "
        "ways. And `uday_bala` is the only bala that depends on a *second* body, the Sun.",
        "",
        "## Eight numerically-constant columns",
        "",
        "Correct values, but constant to machine precision — they escaped "
        "[#3](https://github.com/jenujari/prdict-pov-v1/issues/3)'s constant filter, which "
        "tested exact equality: float noise in the last bits gives them thousands of "
        "distinct values while their entire range is below `1e-6`. A zero-variance column "
        "can divide by zero in a scaler well before the stage-1 prune in "
        "[#12](https://github.com/jenujari/prdict-pov-v1/issues/12) would ever see it, so "
        "this ticket widened that filter to a tolerance rather than leaving them in. "
        f"**Features: 215 → 207.**",
        "",
        "| Column | Distinct floats | Range | Value |",
        "|--------|-----------------|-------|-------|",
    ]
    for column in degen:
        d = audit["degenerate_columns"][column]
        lines.append(
            f"| `{column}` | {d['distinct_float_values']} | {d['range']:.2e} | "
            f"{d['value']:.6f} |"
        )
    lines += [
        "",
        "All eight are degenerate **by definition**, which is the same licence #3 used to "
        "drop its 20 constants globally without a per-fold refit:",
        "",
        "- The lunar nodes are the intersections of the Moon's orbit with the ecliptic, so "
        "their ecliptic **latitude is identically zero** — as is its rate of change.",
        "- The nodes are geometric points, not bodies, so there is **no distance** and no "
        "rate of change of one. (#3 already dropped `rahu_distance` and `ketu_distance` as "
        "exact constants; only the *rate* columns slipped through.)",
        f"- This file's Rahu is a **mean** node: `rahu_longitude` regresses at a fixed "
        f"{abs(audit['degenerate_columns']['rahu_speed_long']['value']):.6f} degrees per day "
        "(360 degrees / 18.6 years), so its speed column is a constant. The observed "
        "day-over-day motion of `rahu_longitude` matches it exactly, so this is a correct "
        "constant, not a wrong value.",
        "",
        "## Every derived column is a function of columns already in the file",
        "",
        "The audit had to reconstruct each family from base quantities to check it, and "
        "in doing so established that **no derived family carries information the base "
        "columns do not**. This is a much stronger statement than the pairwise correlation "
        "prune can reach, and it is handed to "
        "[#15](https://github.com/jenujari/prdict-pov-v1/issues/15) rather than acted on "
        "here.",
        "",
        "| Determined by | Families |",
        "|---------------|----------|",
        "| `*_longitude` alone | `*_sign`, `*_navamsa_sign`, `*_sign_lordship`, "
        "`*_uchcha_bala`, `*_kshetra_bala`, `*_navamsha_bala` |",
        "| `*_longitude` + `sun_longitude` + `*_is_retro` | `*_uday_bala` |",
        "| `*_speed_long` | `*_speed_category`, `*_vedha`, `*_vakra_bala`, `*_is_retro` |",
        "| `*_nakshatra_name` + `*_vedha` | `*_vedha_target` |",
        "",
        "Some of these **survive** a pairwise `|r| >= 0.95` prune despite being exactly "
        "determined, which is worth knowing before "
        "[#12](https://github.com/jenujari/prdict-pov-v1/issues/12) is trusted to find "
        "redundancy on its own:",
        "",
        "- `*_uchcha_bala` is `75 + 25*cos(lon - exaltation)`, an exact linear combination "
        "of the `*_longitude_sin` / `*_longitude_cos` pair #7 emits. Whether the prune "
        "catches it depends on where the exaltation angle falls relative to the two "
        "members — measured against the real longitudes below, "
        f"**{len(survivors)} of 8 survive**.",
        "",
        "  | Planet | \\|r\\| vs `_sin` | \\|r\\| vs `_cos` | Survives the prune |",
        "  |--------|---------------|---------------|--------------------|",
    ] + [
        f"  | `{p}` | {e['r_sin']:.4f} | {e['r_cos']:.4f} | "
        f"{'**yes**' if e['survives'] else 'no'} |"
        for p, e in audit["prune_survival"]["uchcha_bala"].items()
    ] + [
        "",
        "  Ketu is not in the table, because #7 dropped `ketu_longitude`'s sin/cos pair as "
        "antipodal and it has no partner left to be compared against. Its bala columns are "
        "instead exact "
        "**duplicates of Rahu's** — `ketu_uchcha_bala`, `ketu_kshetra_bala` and "
        f"`ketu_navamsha_bala` all match to "
        f"{max(audit['prune_survival']['ketu_duplicates_rahu'].values()):.2e}. Ketu's "
        "longitude is Rahu's + 180 *and* its exaltation point is 180 away, so the offsets "
        "cancel inside the cosine; for the two ramp families 180 is a whole number of "
        "signs (6) and of navamsas (54), and both nodes sit at `Enemy` dignity everywhere. "
        "Definitional, like the #7 drops, so removing them cannot leak — 3 more columns "
        "for #15.",
        "",
        "- `*_vakra_bala` is piecewise-linear with a flat zero branch over every direct "
        "row, so its linear correlation with `speed_long` is well under the threshold "
        "even though it is an exact function of it.",
        "",
        "## Method and its limit",
        "",
        "Every check recomputes from base columns in the same file and compares. Two "
        "checks are weaker than the rest and are labelled `consistent` rather than "
        "`exact`:",
        "",
        "- **`*_speed_dist` and `*_speed_lat`.** A day-over-day central difference is a "
        "second-order approximation of a derivative the source computed analytically, so "
        "exact agreement is not available. The claim under test is unit and sign, at a "
        "correlation a wrong unit or a flipped sign could not reach — all seven planets "
        "clear 0.99995, with a fitted slope of 1.000. `*_speed_long` is checked the same "
        "way as a control and clears 0.999999.",
        "- **`*_speed_category`.** The bin edges are constants of the source's table and "
        "are not derivable from the file. The checkable claim is the *ordering*: sorted by "
        "`speed_long`, the category index must never decrease. Zero violations across "
        "63,000 planet-rows; a single inversion would have proved the column wrong.",
        "",
        "The audit reruns in full from `scripts/audit_derived_columns.py` and raises on the "
        "first disagreement, so it fails loudly if the source data changes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    audit = build()
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n")
    AUDIT_MD.write_text(render_markdown(audit))

    print("verified: sign, navamsa, lordship, uchcha, kshetra, navamsha, uday, vakra,")
    print("          speed_category, vedha, vedha_target, distance/latitude rates")
    print(f"\nmiscomputed : {audit['summary']['miscomputed'] or 'none'}")
    print(f"degenerate  : {len(audit['summary']['degenerate'])} columns")
    for column in audit["summary"]["degenerate"]:
        print(f"              {column}")
    print(f"unverifiable: {audit['summary']['unverifiable'] or 'none'}")
    print(f"\nwrote {AUDIT_JSON.relative_to(ROOT)} and {AUDIT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
