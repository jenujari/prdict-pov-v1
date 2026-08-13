#!/usr/bin/env bash
# Trigger TFT training for #39 — run this manually, monitor independently.
#
# At the fixed budget in kb/tft_spec.json (4-trial search, 12 contexts, 60
# total fits), this is unattended, hours-to-possibly-a-day scale (TFT costs
# orders of magnitude more per fit than XGBoost — see kb/research/
# tft-stack-selection.md and the timing probe run before this was launched
# for real). Does not background itself — use tmux/screen (or `nohup ... &`)
# if you want it to survive a disconnect:
#
#   tmux new -s tft './scripts/run_tft_training.sh'
#   # detach: Ctrl-b d ; reattach later: tmux attach -t tft
#
# While it runs:
#   - The terminal redraws a single-frame status board every ~0.5s: which
#     context is active, its trial/epoch progress, and a tail of recent log
#     lines. Safe to just watch, or detach and check back later.
#   - From another terminal:  tail -f runs/tft_training.log
#
# When it ends (normally, on a failed context, or Ctrl-C):
#   - predictions/tft/*.parquet             — per-context prediction files
#   - kb/tft_run.md                         — winning hyperparameters table
#   - runs/tft_summary.txt                  — plain-text summary; paste this
#                                              back into a Claude Code session
#                                              to pick the work back up
#
# Resumable per context (same granularity as #38): each context checkpoints
# the moment it finishes. If the machine reboots, the process is killed, or
# you just Ctrl-C, run this exact same command again — already-finished
# contexts are detected and skipped. A context that was mid-fit when
# interrupted restarts its search+refit from scratch (a TFT fit's own
# mid-epoch Lightning checkpoints aren't wired into the resume path — only
# whole-context completion is). If kb/tft_spec.json changes in between,
# everything reruns from scratch.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"

echo "Starting TFT training (#39) — see runs/tft_training.log for the plain log."
echo "Ctrl-C to stop early; already-finished contexts are checkpointed and this script"
echo "can simply be rerun later to pick up where it left off."
echo

exec uv run python scripts/train_tft.py
