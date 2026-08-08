# prdict-pov-v1

Predicting a 10-day forward Nifty-50 return vector from Vedic-astrology ephemeris features, comparing three models on a trading-simulation scorecard. This glossary fixes the vocabulary the specs, tickets, and code all share.

## Language

### The window

**Session**:
One row of the trading-day index — a day the NSE was open. All windows are counted in sessions, never calendar days.
_Avoid_: day, trading day (when precision matters), bar

**Origin**:
The last session whose close the model may see — the reference point a prediction is made from. Its index is `i`.
_Avoid_: prediction date, anchor, `t0`

**Past block**:
The **60** sessions up to and including the origin. Feeds the model's encoder.
_Avoid_: lookback, history window, encoder window (except when naming the TFT tensor)

**Future block**:
The **30** sessions strictly after the origin. Every feature in it is astronomical, so it is known in advance from the ephemeris — this is what makes it usable as input.
_Avoid_: forecast window, decoder window (except when naming the TFT tensor), horizon

**Window**:
Past block + future block = **90 sessions** of features. The full input span for one origin. (The headline "120-day" framing was a miscount — see ADR 0001.)
_Avoid_: input tensor, sequence

**Horizon**:
The **10** sessions after the origin that are actually scored. The prediction target spans exactly these.
_Avoid_: future block (that is 30 sessions; the horizon is the first 10 of them)

### The target

**Scored target**:
The `(n, 10)` array of step log returns `r_k = log(C_{i+k}/C_{i+k-1})`, `k = 1..10`. The single quantity every model is compared on.
_Avoid_: label (ambiguous with the training target), y

**Training target**:
For the TFT only, the `(n, 30)` extension of the scored target, needed because the TFT decodes all 30 future sessions (ADR 0002). Only the first 10 steps are ever scored. XGBoost trains directly on the 10-step scored target.
_Avoid_: decoder target, full label

**Elapsed covariate**:
Calendar days spanned by each target step (`elapsed_k`), a known-future decoder-side input — not a member of the 280-column feature block. One time-varying real for the TFT; `elapsed_1..10` columns for XGBoost.
_Avoid_: time delta, gap

### Origins by role

**Trainable origin**:
An origin whose whole target is observed. **6448** for the 10-step scored target (`C_{i+10}` observed); ~**6428** for the TFT's 30-step training target (`C_{i+30}` observed).
_Avoid_: training sample

**Forward origin**:
An origin past the last observed close, predicted at inference. Its features come from the ephemeris; it has no target. **117** of them, bounded by the ephemeris end, not by prices.
_Avoid_: test origin, future origin

### Cross-validation

**Purge**:
Sessions dropped at a fold boundary so no training origin's target overlaps a validation origin's scored days. **30 sessions** (set by the TFT's 30-step training target; ADR 0002).
_Avoid_: gap, buffer

**Embargo**:
The purge applied before the final holdout. Same 30-session value.
_Avoid_: holdout gap

### Inputs and models

**Set 1**:
The correlation-pruned feature set — stage-1 (`|r|≥0.95` redundancy) then stage-2 (mutual-information / Spearman ranking) selection, fit in-fold.
_Avoid_: pruned set, selected features

**Set 2**:
The PCA(0.95)-reduced feature set, preserving the 90-session time axis, fit in-fold.
_Avoid_: PCA set, reduced set

**Baseline**:
The always-predict-up model — the trivial reference the other two must beat on the scorecard.
_Avoid_: naive model, control

**Scorecard**:
The trading-simulation evaluation: positions derived from the predicted return vector, scored on cumulative return, Sharpe, and max drawdown over the holdout.
_Avoid_: metrics, evaluation, benchmark
