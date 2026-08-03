# Column specification — `nft50.csv`

Generated 2026-08-03 by `scripts/build_column_spec.py`. Do not hand-edit — rerun the script.

Resolves [#3](https://github.com/jenujari/prdict-pov-v1/issues/3).

`nft50.csv`: **240 columns**, 9893 rows. Labels (`c`) run to **2026-06-30**.

## Roles

| Role | Count | Meaning |
|------|-------|---------|
| `key` | 1 | The date index. |
| `target_source` | 1 | Close price — source of the label, never an input (map decision 5). |
| `dropped_price` | 3 | Open/high/low, dropped with all price-derived features (map decision 5). |
| `dropped_constant` | 28 | Single-valued across the file. All structural — see below. |
| `feature` | 207 | Model inputs. |

## Types

| Type | Count |
|------|-------|
| `boolean` | 14 |
| `categorical` | 67 |
| `angular` | 53 |
| `linear_numeric` | 73 |
| **total** | **207** |

### `boolean`

14 columns, all `*_is_retro` or `*_vargottama`. Note these are also declared in `categories_list.json`; the bool dtype wins, so they are typed once, here.

- `sun_vargottama`
- `moon_vargottama`
- `saturn_is_retro`
- `saturn_vargottama`
- `venus_is_retro`
- `venus_vargottama`
- `mercury_is_retro`
- `mercury_vargottama`
- `jupiter_is_retro`
- `jupiter_vargottama`
- `rahu_vargottama`
- `ketu_vargottama`
- `mars_is_retro`
- `mars_vargottama`

### `categorical`

67 columns. Levels come from `categories_list.json` — the **full declared list**, never the observed values, so that encoding is stable across CV folds (map decision 6).

- `weekday`
- `sun_speed_category`
- `sun_vedha`
- `sun_vedha_target`
- `sun_sign`
- `sun_nakshatra_name`
- `sun_sign_lord`
- `sun_sign_lordship`
- `sun_navamsa_sign`
- `moon_speed_category`
- `moon_vedha`
- `moon_vedha_target`
- `moon_sign`
- `moon_nakshatra_name`
- `moon_sign_lord`
- `moon_sign_lordship`
- `moon_navamsa_sign`
- `saturn_speed_category`
- `saturn_vedha`
- `saturn_vedha_target`
- `saturn_sign`
- `saturn_nakshatra_name`
- `saturn_sign_lord`
- `saturn_sign_lordship`
- `saturn_navamsa_sign`
- `venus_speed_category`
- `venus_vedha`
- `venus_vedha_target`
- `venus_sign`
- `venus_nakshatra_name`
- `venus_sign_lord`
- `venus_sign_lordship`
- `venus_navamsa_sign`
- `mercury_speed_category`
- `mercury_vedha`
- `mercury_vedha_target`
- `mercury_sign`
- `mercury_nakshatra_name`
- `mercury_sign_lord`
- `mercury_sign_lordship`
- `mercury_navamsa_sign`
- `jupiter_speed_category`
- `jupiter_vedha`
- `jupiter_vedha_target`
- `jupiter_sign`
- `jupiter_nakshatra_name`
- `jupiter_sign_lord`
- `jupiter_sign_lordship`
- `jupiter_navamsa_sign`
- `rahu_vedha_target`
- `rahu_sign`
- `rahu_nakshatra_name`
- `rahu_sign_lord`
- `rahu_navamsa_sign`
- `ketu_vedha_target`
- `ketu_sign`
- `ketu_nakshatra_name`
- `ketu_sign_lord`
- `ketu_navamsa_sign`
- `mars_speed_category`
- `mars_vedha`
- `mars_vedha_target`
- `mars_sign`
- `mars_nakshatra_name`
- `mars_sign_lord`
- `mars_sign_lordship`
- `mars_navamsa_sign`

### `angular`

53 columns. This spec fixes **membership only**. Each column's period and its sin/cos transform are decided in [#7](https://github.com/jenujari/prdict-pov-v1/issues/7).

- `tithy`
- `sun_longitude`
- `sun_nakshatra_pada`
- `moon_longitude`
- `moon_nakshatra_pada`
- `saturn_longitude`
- `saturn_nakshatra_pada`
- `venus_longitude`
- `venus_nakshatra_pada`
- `mercury_longitude`
- `mercury_nakshatra_pada`
- `jupiter_longitude`
- `jupiter_nakshatra_pada`
- `rahu_longitude`
- `rahu_nakshatra_pada`
- `ketu_longitude`
- `ketu_nakshatra_pada`
- `mars_longitude`
- `mars_nakshatra_pada`
- `sun_mars_dist`
- `sun_mercury_dist`
- `sun_jupiter_dist`
- `sun_venus_dist`
- `sun_saturn_dist`
- `sun_rahu_dist`
- `sun_ketu_dist`
- `moon_mars_dist`
- `moon_mercury_dist`
- `moon_jupiter_dist`
- `moon_venus_dist`
- `moon_saturn_dist`
- `moon_rahu_dist`
- `moon_ketu_dist`
- `mars_mercury_dist`
- `mars_jupiter_dist`
- `mars_venus_dist`
- `mars_saturn_dist`
- `mars_rahu_dist`
- `mars_ketu_dist`
- `mercury_jupiter_dist`
- `mercury_venus_dist`
- `mercury_saturn_dist`
- `mercury_rahu_dist`
- `mercury_ketu_dist`
- `jupiter_venus_dist`
- `jupiter_saturn_dist`
- `jupiter_rahu_dist`
- `jupiter_ketu_dist`
- `venus_saturn_dist`
- `venus_rahu_dist`
- `venus_ketu_dist`
- `saturn_rahu_dist`
- `saturn_ketu_dist`

### `linear_numeric`

73 columns — latitudes (a narrow ±8.6 band, not cyclic), distances in AU, the three speed families, and the bala scores.

- `sun_latitude`
- `sun_distance`
- `sun_speed_long`
- `sun_speed_lat`
- `sun_speed_dist`
- `sun_uchcha_bala`
- `sun_kshetra_bala`
- `sun_navamsha_bala`
- `moon_latitude`
- `moon_distance`
- `moon_speed_long`
- `moon_speed_lat`
- `moon_speed_dist`
- `moon_uday_bala`
- `moon_uchcha_bala`
- `moon_kshetra_bala`
- `moon_navamsha_bala`
- `saturn_latitude`
- `saturn_distance`
- `saturn_speed_long`
- `saturn_speed_lat`
- `saturn_speed_dist`
- `saturn_uday_bala`
- `saturn_uchcha_bala`
- `saturn_vakra_bala`
- `saturn_kshetra_bala`
- `saturn_navamsha_bala`
- `venus_latitude`
- `venus_distance`
- `venus_speed_long`
- `venus_speed_lat`
- `venus_speed_dist`
- `venus_uday_bala`
- `venus_uchcha_bala`
- `venus_vakra_bala`
- `venus_kshetra_bala`
- `venus_navamsha_bala`
- `mercury_latitude`
- `mercury_distance`
- `mercury_speed_long`
- `mercury_speed_lat`
- `mercury_speed_dist`
- `mercury_uday_bala`
- `mercury_uchcha_bala`
- `mercury_vakra_bala`
- `mercury_kshetra_bala`
- `mercury_navamsha_bala`
- `jupiter_latitude`
- `jupiter_distance`
- `jupiter_speed_long`
- `jupiter_speed_lat`
- `jupiter_speed_dist`
- `jupiter_uday_bala`
- `jupiter_uchcha_bala`
- `jupiter_vakra_bala`
- `jupiter_kshetra_bala`
- `jupiter_navamsha_bala`
- `rahu_uchcha_bala`
- `rahu_kshetra_bala`
- `rahu_navamsha_bala`
- `ketu_uchcha_bala`
- `ketu_kshetra_bala`
- `ketu_navamsha_bala`
- `mars_latitude`
- `mars_distance`
- `mars_speed_long`
- `mars_speed_lat`
- `mars_speed_dist`
- `mars_uday_bala`
- `mars_uchcha_bala`
- `mars_vakra_bala`
- `mars_kshetra_bala`
- `mars_navamsha_bala`

## Availability: known-future vs past-only

Map decision 3 called for splitting covariates into known-future and past-only. With price features dropped (decision 5), **every one of the 207 feature columns is `known_future`** — the ephemeris is computed, so it is fully populated through the end of the file. The build script asserts this rather than assuming it.

The practical consequence: the past-60 and future-30 blocks carry an **identical feature set**, so the 90-step window is simply contiguous and the encoder/decoder boundary is a slicing convention. This is the input [#11](https://github.com/jenujari/prdict-pov-v1/issues/11) was told to verify.

## Structural nulls

`sun_vedha_target` is the only feature column with missing values (2685 rows, 55 of them after the label cutoff). The nulls are **structural, not missing**: the column is null exactly when `sun_vedha == 'no'`, a perfect 1:1 match that the build script asserts. No vedha in effect means there is no target nakshatra to name.

Handled by appending a `"none"` sentinel to the declared level list and filling with it — not by imputation, which would invent a nakshatra that astronomically is not there.

## Constant columns

All 28 constant columns are constant **by definition**, not by sampling accident. That is what licenses dropping them once, globally, rather than re-deciding inside every CV fold — the concern raised in [#10](https://github.com/jenujari/prdict-pov-v1/issues/10).

| Column | Why constant |
|--------|--------------|
| `ketu_distance` | The nodes are geometric points, not bodies — no physical distance exists. |
| `ketu_is_retro` | Ketu is a computed lunar node; it is always retrograde by convention. |
| `ketu_latitude` | A lunar node lies on the ecliptic by definition, so its latitude is identically zero (#30). |
| `ketu_sign_lordship` | Declared with two levels but only 'Enemy' occurs. Same full-cycle-coverage argument as rahu_sign_lordship. |
| `ketu_speed_category` | Follows from ketu_is_retro — a permanently retrograde body is always 'vakra'. |
| `ketu_speed_dist` | Rate of change of a distance that does not exist; zero to machine precision (#30). |
| `ketu_speed_lat` | Rate of change of a latitude that is identically zero (#30). |
| `ketu_speed_long` | Ketu is antipodal to Rahu, so it inherits the mean node's fixed rate (#30). |
| `ketu_uday_bala` | Formula constant for the nodes. |
| `ketu_vakra_bala` | Retrograde strength is maximal for a permanently retrograde body. |
| `ketu_vedha` | Vedha is not defined for the shadow planets; the source emits a fixed placeholder. |
| `moon_is_retro` | The Moon never appears retrograde from Earth. |
| `moon_vakra_bala` | Retrograde strength is zero for a body that never retrogrades. |
| `rahu_distance` | The nodes are geometric points, not bodies — no physical distance exists. |
| `rahu_is_retro` | Rahu is a computed lunar node; it is always retrograde by convention. |
| `rahu_ketu_dist` | Rahu and Ketu are antipodal by construction, so their separation is always 180 degrees. |
| `rahu_latitude` | A lunar node lies on the ecliptic by definition, so its latitude is identically zero (#30). |
| `rahu_sign_lordship` | Declared with two levels but only 'Enemy' occurs. Rahu traverses all twelve signs roughly every 18.6 years, so 27 years of history covers the full cycle many times over — the unobserved level is unreachable, not merely unsampled. |
| `rahu_speed_category` | Follows from rahu_is_retro — a permanently retrograde body is always 'vakra'. |
| `rahu_speed_dist` | Rate of change of a distance that does not exist; zero to machine precision (#30). |
| `rahu_speed_lat` | Rate of change of a latitude that is identically zero (#30). |
| `rahu_speed_long` | This file's Rahu is a *mean* node: rahu_longitude regresses at a fixed 0.052992 degrees per day (360 degrees / 18.6 years), matching the observed day-over-day motion exactly, so the speed column is a constant (#30). |
| `rahu_uday_bala` | Formula constant for the nodes. |
| `rahu_vakra_bala` | Retrograde strength is maximal for a permanently retrograde body. |
| `rahu_vedha` | Vedha is not defined for the shadow planets; the source emits a fixed placeholder. |
| `sun_is_retro` | The Sun never appears retrograde from Earth. |
| `sun_uday_bala` | Formula constant for the Sun. |
| `sun_vakra_bala` | Retrograde strength is zero for a body that never retrogrades. |

## Exact derived relationships

Deterministic dependencies between feature columns. These matter because the stage-1 redundancy prune in [#12](https://github.com/jenujari/prdict-pov-v1/issues/12) works on pairwise `|r|` and **will not catch them** — a nonlinear function of a difference of two columns has no strong linear correlation with either input.

### `tithy`

```
tithy = floor(((moon_longitude - sun_longitude) mod 360) / 12) + 1
```

Exact on 100% of rows. This is also why there is no sun_moon_dist column among the 35 cross-planet separations — tithy is that separation, binned into 30 steps of 12 degrees.

## Declared levels never observed

These levels appear in `categories_list.json` but never in 27 years of data. They are retained in the encoding anyway (map decision 6) so that category codes stay stable and an unseen level at inference time does not shift every other code.

| Column | Never observed |
|--------|----------------|
| `sun_speed_category` | `kutil`, `ati-vakra`, `vakra` |
| `sun_vedha` | `right` |
| `sun_vedha_target` | `Krittika`, `Rohini`, `Hasta`, `Chitra`, `Swati`, `Vishakha`, `Uttara Ashadha`, `Abhijit` |
| `sun_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `moon_speed_category` | `kutil`, `ati-vakra`, `vakra` |
| `moon_vedha` | `right`, `no` |
| `moon_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `moon_sign_lordship` | `Enemy` |
| `saturn_vedha` | `no` |
| `saturn_nakshatra_name` | `Ashwini` |
| `saturn_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `venus_vedha` | `no` |
| `venus_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `mercury_vedha` | `no` |
| `mercury_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `jupiter_vedha` | `no` |
| `jupiter_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `rahu_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `ketu_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |
| `mars_vedha` | `no` |
| `mars_sign_lord` | `Uranus`, `Neptune`, `Pluto`, `Rahu`, `Ketu` |

## `log1p` is unsafe here

`backup.one.txt` applied `log1p` to any feature with `|skew| > 1.5`. The following columns reach `-1` or below, where `log1p` returns NaN:

- `jupiter_latitude`
- `mars_latitude`
- `mercury_latitude`
- `mercury_speed_long`
- `moon_latitude`
- `moon_speed_lat`
- `saturn_latitude`
- `venus_latitude`

Signed rate-of-change columns must not be log-transformed. Carried into [#9](https://github.com/jenujari/prdict-pov-v1/issues/9), which decides skew handling.
