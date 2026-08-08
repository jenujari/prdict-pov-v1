# Input window is 60 past + 30 future sessions; the "120-day" framing is rejected

The goal was once restated as a "120-day input feature tensor," and `audit-tickets-2-10.md` (D1/D2) treated the repo's 60-session past block as a divergence to be fixed by re-windowing to 120. That is wrong: the window is **60 past + 30 future = 90 sessions** counted in trading days, and predicts a **10-session** horizon. The "120" was arithmetic drift (60 + 30 = 90, not 120), confirmed with the author. The existing calendar, target, and fold specs — all built on 60/30/10 — are therefore **correct as-is**, and the audit's D1/D2 and their eight-step "fix" are void.

## Consequences

- No re-windowing. `PAST=60`, `FUTURE=30`, `HORIZON=10` in `scripts/build_trading_calendar.py` stay; the `past-60 / future-30 / 90-step` language in the kb specs is accurate, not stale.
- The 30-session future block is kept deliberately, as *input*: 30 days of known-future planetary positions feed the models even though only 10 days are scored. This is design intent, not waste — it is the reason the audit's D2 (shrink future to 10) is rejected.
- Recorded because a future reader (and one existing repo document) will otherwise re-open the "should this be 120?" question.
