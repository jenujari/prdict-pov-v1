# Walk-Forward CV & Holdout Specification

Generated 2026-08-03 by `scripts/build_fold_spec.py`. Do not hand-edit.
Resolves [#10](https://github.com/jenujari/prdict-pov-v1/issues/10).

## Summary

- **Total Trainable Origins**: `6448` (2000-03-29 → 2026-06-15)
- **CV Geometry**: Expanding Window (Anchored Walk-Forward)
- **Validation Folds**: 5 folds (2010-04-01 → 2023-12-29, `3393` origins)
- **Final Holdout**: `601` origins (`2024-01-01` → `2026-06-15`)

## 1. Walk-Forward Fold Boundaries

| Fold | Train Range | Train Origins | Validation Range | Val Origins | Purge Pre | Embargo Post |
|------|-------------|---------------|------------------|-------------|-----------|--------------|
| Fold 1 | `2000-03-29` → `2010-03-17` | 2445 | `2010-04-01` → `2012-12-19` | 678 | 10 s | 69 s |
| Fold 2 | `2000-03-29` → `2012-12-06` | 3123 | `2012-12-20` → `2015-09-22` | 678 | 10 s | 69 s |
| Fold 3 | `2000-03-29` → `2015-09-08` | 3801 | `2015-09-23` → `2018-06-22` | 678 | 10 s | 69 s |
| Fold 4 | `2000-03-29` → `2018-06-11` | 4479 | `2018-06-25` → `2021-03-24` | 678 | 10 s | 69 s |
| Fold 5 | `2000-03-29` → `2021-03-10` | 5157 | `2021-03-25` → `2023-12-29` | 681 | 10 s | 69 s |

## 2. Final Holdout Set

- **Range**: `2024-01-01` → `2026-06-15` (601 trainable origins)
- **Policy**: Strictly untouched final holdout set containing post-2023 market regimes.

## 3. Purge & Embargo Arithmetic

| Parameter | Length | Arithmetic & Derivation |
|-----------|--------|--------------------------|
| **Purge (Pre-Validation)** | `10 sessions` | 10 sessions (target horizon) to eliminate direct target label leakage from train into val. |
| **Embargo (Post-Validation)** | `69 sessions` | 69 sessions (60 encoder lookback + 9 label overlap) for 100% complete row-level independence. |

## 4. Explicit In-Fold Fits

To prevent lookahead leak, the following transformations MUST be fitted strictly inside each training fold:

| Stage | Description |
|-------|-------------|
| **Scaler** | StandardScaler / RobustScaler fitted strictly on training fold origins. |
| **Correlation Prune** | Pairwise correlation matrix (|r| >= 0.95) computed strictly on training fold origins. |
| **Feature Ranking** | Mutual Information / Spearman correlation ranking computed strictly on training fold origins. |
| **PCA** | Principal Component Analysis (n_components=0.95) fitted strictly on scaled training fold features. |
| **Data-Dependent Redundancy / Variance Filter** | Any empirical zero-variance or empirical redundancy filtering evaluated strictly inside each training fold. |

## 5. Overlapping-Sample Dependence & Reporting

- **Sample Dependence**: Consecutive origins share 9 of 10 target label days. This autocorrelation increases metric variance across folds.
- **Reporting Policy**: Out-of-sample predictions across 5 validation folds (3393 origins) concatenated into one continuous time-series for trading simulation.
- **Summary Metrics**: Per-fold metrics (Sharpe, max drawdown, win rate) reported as mean +- std across all 5 folds.
