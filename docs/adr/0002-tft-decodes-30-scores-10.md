# The TFT decodes 30 sessions but is scored on 10, fixing the fold purge at 30

`pytorch-forecasting`'s TFT ties the known-future covariate span to `max_prediction_length`: it only ingests future covariates for as many steps as it decodes. Since the design feeds **30** future sessions as input (ADR 0001) but scores a **10**-session horizon, we set `max_encoder_length=60`, `max_prediction_length=30`, train against a real 30-step target built from history, and **use only the first 10 predicted steps** for the trading simulation and the cross-model comparison. This closes #5's open "10 vs 30 decoder steps" question at **30** — the opposite of the audit's guess — because the intent is 30-in.

## Considered options

- **Decoder length 10** — simplest and matches the scored target exactly, but the TFT would then see only 10 future days while XGBoost (flat matrix) sees all 30, confounding the comparison and blinding the TFT to 20 days of the exact planetary context the hypothesis is about. Rejected.
- **Full bypass** — instantiate `TemporalFusionTransformer` directly with a custom `DataLoader` feeding 30 decoder-covariate steps but a 10-step loss. Most faithful, but off the supported path and a hand-rolled training loop. Rejected as unnecessary given the option above works on supported rails.

## Consequences

- **The scored target stays `(n, 10)`.** #8's definition is untouched; the 30-step target is a training-only extension for the TFT decoder (`y_1..y_30`, `elapsed_1..30`).
- **TFT trainable origins ≈ 6428** (needs `C_{i+30}` observed) vs 6448 for XGBoost; the ~20 lost origins sit in the final training window and are negligible. Scoring/holdout origins are identical for both models.
- **Fold purge rises from 10 to 30.** A 30-step training label reaches 30 sessions forward, so the label-overlap floor that governs purge/embargo is 30, not 10. Both models use the single 30-session geometry so the comparison runs on identical folds; the extra cost to XGBoost is ~20 origins per boundary (~0.3%). This reopens #10 for a numbers-only rebuild.
