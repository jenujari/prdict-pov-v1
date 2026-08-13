#!/usr/bin/env bash
# Trigger the #54 significance test — run this manually, monitor independently.
#
# At the fixed budget in kb/significance_spec.json (N=200 permutations, one
# XGBoost fit each on set2's real winning config), this is a single-context
# loop — much lighter than #38/#39's multi-context runs. Timing-probed at
# ~27s/permutation, so a full run is on the order of 1.5 hours, not
# overnight. Does not background itself — use tmux/screen (or `nohup ... &`)
# if you want it to survive a disconnect:
#
#   tmux new -s significance './scripts/run_significance_test.sh'
#   # detach: Ctrl-b d ; reattach later: tmux attach -t significance
#
# While it runs:
#   - Plain log to stdout and runs/significance_test.log — no dashboard here,
#     the loop is homogeneous (one context, N trials) so a scrolling log is
#     as legible as a redrawing status board would be.
#   - From another terminal:  tail -f runs/significance_test.log
#
# When it ends (normally, or Ctrl-C):
#   - kb/significance_test.md          — the p-value and null distribution,
#                                          written only once all N permutations
#                                          are complete
#   - runs/significance_checkpoint.parquet — the resume checkpoint
#
# Resumable: each completed permutation checkpoints immediately. If
# interrupted, rerun this exact same command — already-finished permutations
# are detected and skipped. If kb/significance_spec.json changes in between,
# everything reruns from scratch (a checkpoint only counts under the budget
# it was actually produced with).
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"

echo "Starting significance test (#54) — see runs/significance_test.log for the plain log."
echo "Ctrl-C to stop early; already-finished permutations are checkpointed and this script"
echo "can simply be rerun later to pick up where it left off."
echo

exec uv run python scripts/run_significance_test.py
