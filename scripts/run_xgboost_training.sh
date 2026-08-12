#!/usr/bin/env bash
# Trigger XGBoost training for #38 — run this manually, monitor independently.
#
# At the fixed budget in kb/xgboost_spec.json (16 trials, max_boost_round=500,
# 12 contexts), a full run is several hours; realistically dominated by set 2's
# wider (4,420-column) matrix. This does not background itself — it runs in the
# foreground of whatever terminal you launch it from, so use tmux/screen (or
# `nohup ... &`) if you want it to survive a disconnect:
#
#   tmux new -s xgboost './scripts/run_xgboost_training.sh'
#   # detach: Ctrl-b d ; reattach later: tmux attach -t xgboost
#
# While it runs:
#   - The terminal redraws a single-frame status board every ~0.5s: which
#     context is active, its trial/round progress, and a tail of recent log
#     lines. Safe to just watch, or detach and check back later.
#   - From another terminal:  tail -f runs/xgboost_training.log
#
# When it ends (normally, on a failed context, or Ctrl-C):
#   - predictions/xgboost/*.parquet         — per-context prediction files
#   - kb/xgboost_run.md                     — winning hyperparameters table
#   - runs/xgboost_summary.txt              — plain-text summary; paste this
#                                              back into a Claude Code session
#                                              to pick the work back up
#
# Resumable: each context checkpoints (predictions + a small metadata sidecar)
# the moment it finishes. If the machine reboots, the process is killed, or you
# just Ctrl-C, run this exact same command again — already-finished contexts
# are detected and skipped, so you only pay for whatever was still pending.
# (If kb/xgboost_spec.json changes in between, everything reruns from scratch —
# a checkpoint only counts under the budget it was actually produced with.)
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"

echo "Starting XGBoost training (#38) — see runs/xgboost_training.log for the plain log."
echo "Ctrl-C to stop early; already-finished contexts are checkpointed and this script"
echo "can simply be rerun later to pick up where it left off."
echo

exec uv run python scripts/train_xgboost.py
