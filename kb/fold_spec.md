# Walk-forward folds, holdout, and purge arithmetic

Generated 2026-08-12 by `scripts/build_fold_spec.py`. Do not hand-edit.
Resolves [#10](https://github.com/jenujari/prdict-pov-v1/issues/10).

## Summary

- **Geometry**: rolling window — every fold trains on exactly `2470` origins (~10 trading years).
- **Folds**: 5, validating `3347` origins from `2010-06-08` to `2023-12-29`.
- **Purge**: `30` sessions before every validation block. No embargo.
- **Holdout**: `601` origins, `2024-01-01` → `2026-06-15`.
- **Total trainable origins**: `6448`.

## 1. Why rolling, not expanding

Under an expanding window the last fold trains on roughly twice the data of the
first. A fold-to-fold difference in score then confounds two causes — the regime
changed, or the model simply had more data — and the concatenated series is
produced by five models of systematically increasing capability. Holding the
training length fixed removes that confound, so fold-to-fold differences are about
regime alone. The cost is recorded in §7: the later folds drop the early history,
so fold 5 never sees 2008.

## 2. Derivation

Nothing below is a hand-picked date. Three choices are stated; every boundary
follows from them and the trading calendar.

| Quantity | Value | Source |
|---|---|---|
| Sessions per trading year | 247 | median over complete observed years |
| Training years | 10 | **chosen** — long enough to span the 2000-01 bear, the 2003-07 bull and a crash |
| Training length | 2470 origins | derived: 10 x 247 |
| Folds | 5 | **chosen** — see the effective sample size in §6 |
| Validation per fold | 669 origins | derived from the remainder |
| CV pool | 5847 origins | trainable origins before the holdout |

`5847 CV origins = 2470 train + 30 purge + 5 x 669 validation + 2 remainder`

## 3. Folds

| Fold | Inner train | Inner val | Validation |
|---|---|---|---|
| 1 | `2000-03-30` → `2008-08-18` (2071) | `2008-10-07` → `2010-04-27` (370) | `2010-06-08` → `2013-02-11` (669) |
| 2 | `2002-12-13` → `2011-05-20` (2071) | `2011-07-01` → `2013-01-01` (370) | `2013-02-12` → `2015-11-04` (669) |
| 3 | `2005-08-12` → `2014-01-31` (2071) | `2014-03-19` → `2015-09-21` (370) | `2015-11-05` → `2018-07-19` (669) |
| 4 | `2008-04-30` → `2016-11-01` (2071) | `2016-12-14` → `2018-06-08` (370) | `2018-07-20` → `2021-04-09` (669) |
| 5 | `2011-02-16` → `2019-07-17` (2071) | `2019-08-30` → `2021-02-24` (370) | `2021-04-12` → `2023-12-29` (671) |

Inner train plus inner val is the whole training block for that fold (`2470` origins, less the inner purge). Every fold's train-to-validation separation is exactly `30` sessions.

## 4. Purge

**30 sessions, before every validation block.**

A sample at origin t carries labels for the closes at t+1..t+k. The scored target is 10 steps, but the TFT trains on the full 30-step future block (#8 as amended by #36, docs/adr/0002), so the longest training label reaches t+30. Two origins separated by fewer than 30 sessions therefore share at least one training-label return. Separation of exactly 30 is the floor at which no training label — 10-step or 30-step — and no validation scored label share a return. Both models run on this one geometry: for the 10-step XGBoost the extra 20 sessions cost ~20 origins per boundary (~0.3%), which buys a clean shared comparison.

**Known residual.** At 30 sessions the separation is label-level, not row-level: no training label, 10-step or 30-step, shares a scored return with any validation origin. The last training origin's 30-session known-future covariate block still reaches up to the first validation origin's own encoder rows, so the scaler and PCA see rows carrying near-validation dates. Accepted deliberately: every one of those columns is astro or calendar (#3 — every feature is known-future), so the rows hold no information that would not have been available at the training origin. Full row-level separation would additionally cost the 60-session encoder overlap (89 sessions in total) and is not bought, on the same argument PR #34 used to drop the embargo. Recorded so the choice is visible rather than implied.

**No embargo.** An embargo withholds training rows that sit *after* a validation block. Under any forward walk-forward geometry — rolling or expanding — every training block ends before its own validation block begins, so there are no such rows to withhold. An embargo only does work in interleaved k-fold CV, where a test block has training data on both sides. The previous revision of this spec recorded a 69-session embargo; it withheld nothing, and all 69 of fold 1's nominally embargoed origins were present in the training sets of folds 3, 4 and 5.

## 5. Holdout

- **Range**: `2024-01-01` → `2026-06-15` (601 origins).
- **Final training block**: `2013-11-05` → `2023-11-16` (2470 origins), purged by `30` sessions from the first holdout origin — the same rolling geometry and the same purge as every fold.
- **Final inner validation**: `2022-05-18` → `2023-11-16`.
- **Policy**: Untouched until all three models are final and compared once. The models scored on it are refit on the final training block above — the same rolling geometry, the same purge — so the holdout comparison is not a different experiment from the folds.

**Known limitation.** The window is a calm rising regime: +9.8% total, 13.9% annualised volatility, worst drawdown -15.8%, worst single day -6.1%. The history it is drawn from contains far harsher regimes (2008: -51.8%, 45.8% vol, -59.4% drawdown; 2020: -38.4% drawdown, -13.9% worst day). Because the always-up baseline scores well on a rising window by construction, this holdout has limited power to separate the astro models from that baseline on drawdown or Sharpe. Chosen deliberately to keep the CV pool large; the write-up must state that the comparison is evidenced on one benign regime only.

## 6. Reporting

- **Headline**: One concatenated out-of-sample series. The five validation blocks are contiguous and disjoint, so their predictions join into a single 3347-origin series spanning 2010-06-08 to 2023-12-29, scored once on the trading-simulation scorecard (map decision 8). This is the headline number.
- **Secondary**: Per-fold metrics reported as mean +- std across the five folds, as a consistency check on the headline. A headline carried by one fold is reported as such.
- **Fold boundaries**: Each fold is a separately fitted model, so the concatenated series crosses four model changes. Positions are flattened at each fold boundary and reopened by the incoming model, and the transaction costs of doing so are charged. Carrying a position across a model change would attribute one model's entry to another model's exit.
- **Effective sample size**: Consecutive origins share 9 of 10 label days, so the 3347-origin series is worth roughly 334 independent observations. This does not change the purge — the purge is fixed by label overlap at the fold edge, not by dependence within a block — but it governs interpretation: confidence intervals and any significance test must use the effective count, not the origin count.

### Model selection

The outer validation origins carry the reported out-of-sample trading scorecard. If early stopping or hyperparameter choice also read them, the reported number becomes a best-of-N selected on the very days it is reported for. Overlapping labels make this acute: 9 of every 10 label days are shared, so a fold's validation block is worth roughly 66 independent observations, and best-of-N on that few will manufacture an edge out of noise.

**Rule.** The last 15% of each training block is the inner validation set. The inner training block is everything before it, less the same 30-session purge. Early stopping and hyperparameter selection see the inner validation set and nothing else.

**Budget.** The hyperparameter search space and the number of configurations tried are fixed in writing before any model runs, and are not expanded after seeing inner validation results.

## 7. Gap audit (#25)

- Discontinuities on the index: **28** — 2008's 21 missing sessions (#25), treated as ordinary trading holidays.
- Validation boundaries within 89 sessions of one: **none**
- Training boundaries within 89 sessions of one: **2**
- No validation block contains or comes within 89 sessions of a discontinuity, so no scored origin is affected and no fold boundary needs to move. #25's policy stands: nothing is filtered.

Only training-block edges land near the 2008 stretch, and only because the rolling length places them there — no boundary was chosen to avoid or hit a gap. Fold 4's training block opens inside the defective run, so it carries a partial 2008. Under #25 that is not a defect to correct: the missing sessions are ordinary trading holidays, and a training block that starts mid-year is no more unusual than one that starts mid-quarter.

Under the rolling geometry fold 5 trains from 2011-02-16 and therefore never sees 2008 or 2000-01; fold 4 picks 2008 up only from 2008-04-30. Combined with a holdout drawn from a calm regime, the later folds and the final comparison are trained and scored on progressively calmer data. This is the price of holding the training size constant, and it is recorded rather than hidden.

## 8. What is fitted where

### Fitted inside every fold, on the inner training block only

| Stage | Detail |
|---|---|
| **Scaler** | StandardScaler / RobustScaler, fitted on the inner training block only, so that the inner validation set is scaled by statistics that never saw it. |
| **Correlation prune** | Pairwise \|r\| >= 0.95 among features (map decision 7, stage 1). Refit per fold, and bound by the pair-atomicity constraint below. |
| **Feature ranking** | Mutual information + Spearman against the target, keep top-N (map decision 7, stage 2). Target-dependent, so it must never see validation labels. |
| **PCA** | Per-timestep, numeric-only, n_components=0.95 (map decision 9), fitted on the scaled inner training block. |
| **Empirical variance / redundancy filters** | Any filter whose decision depends on values observed in this fold — near-zero variance, empirical duplicate detection. Distinct from the structural drops below. |

### Applied once, globally — and why that cannot leak

| Stage | Detail |
|---|---|
| **Structural constant drop** | The 28 constant columns from #3 are dropped once, globally. `kb/column_spec.md` records that all 28 are constant *by definition* — a formula constant for the nodes, a mean-node rate, and so on — not constant by sampling accident. A definitional constant carries no fold-specific information, so dropping it globally cannot leak. This is the concern #10 originally raised, and #3 answers it. |
| **Structural Ketu drops** | `ketu_longitude` and the duplicate Ketu balas (#7, #30) are dropped globally on the same argument: they are exact functions of their Rahu twins on every row, definitionally rather than empirically. |
| **Categorical level list** | Levels come from `categories_list.json` — the full declared list, never the observed values (map decision 6). Fitting an encoder per fold would give different embedding widths per fold; the declared list makes the encoding fold-invariant by construction. |
| **Angular sin/cos transform** | Deterministic and parameter-free (#7). Nothing is estimated, so there is nothing to fit inside a fold. |

The distinction is *structural versus empirical*, not *cheap versus expensive*. A
column dropped because a formula makes it constant carries no fold-specific
information and may be dropped once. A column dropped because it happened to look
constant in this sample must be re-decided in every fold.

These map onto [#9](https://github.com/jenujari/prdict-pov-v1/issues/9)'s fit scopes: structural = `none` + `global`, empirical = `fold`. `prdict/encoding.py::assert_fit_boundary()` enforces the encoding-stage half of this split at import time.

### Constraints binding the in-fold stages

**Sin/cos pair atomicity.** The correlation prune and the ranking stage treat each `<col>_sin` / `<col>_cos` pair as one indivisible unit: both survive or both are dropped, and the pair is ranked on its joint score.

*Why:* #7 settled that a lone half is a wrong encoding, not merely a weak one — sin alone folds two configurations onto one value. Because the prune is refit per fold, an unconstrained prune could keep a pair in fold 1 and split it in fold 4, giving the concatenated out-of-sample series several different feature semantics. #7 binds #12 and #13 to this; the same binding applies here, because the in-fold prune is the same operation.

**No target transform is fitted.** The 10-step log-return target passes through untransformed in every fold.

*Why:* #8 rejected winsorisation — the extreme days are the P&L under map decision 8 — so no target statistic is estimated anywhere and the target cannot leak across a fold edge.
