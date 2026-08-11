"""Train XGBoost on both input sets, across every fold and the final holdout refit.

Resolves wayfinder ticket #38. Reads the fixed training contract from
`kb/xgboost_spec.json` (`scripts/build_xgboost_spec.py`, committed separately and
first) and `prdict.xgboost_model`'s matrix builders and search logic. This script
is only the orchestration: which (fold-or-final, set) contexts to run, where
their output lands, and — because a run at the full budget is a multi-hour,
unattended job — a live single-frame terminal dashboard plus a plain log file so
it can be launched and monitored independently of this session.

12 contexts — 5 folds x {set1, set2} (search + refit on `fold.train`, predict on
`fold.val`) plus the final holdout arm x {set1, set2} (search + refit on
`final_train`, predict on `holdout`) — each running the fixed 16-trial search
from `kb/xgboost_spec.json`, ~204 total XGBoost fits. A context that raises is
logged and skipped rather than aborting the whole run; the rest still complete.

Output:
  predictions/xgboost/{fold1..5,final}_{set1,set2}.parquet
      origin date index, y_pred_1..y_pred_10 — #14's scorecard input.
  kb/xgboost_run.md
      winning hyperparameters, inner-val rmse, and timing per context — the
      "run is reproducible" half of #38's done-when.
  runs/xgboost_training.log
      plain timestamped log, written incrementally — `tail -f` this from
      another terminal if you'd rather not watch the redrawing dashboard.
  runs/xgboost_summary.txt
      a short, plain-text summary written when the run ends (success, partial
      failure, or interrupt) — meant to be pasted back into a Claude Code
      session to pick the work back up.

Run via ./scripts/run_xgboost_training.sh, or directly:
    uv run python scripts/train_xgboost.py
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

from prdict import encoding
from prdict.dataset import FlatMatrix
from prdict.folds import load_fold_spec
from prdict.trading_calendar import load_calendar
from prdict.xgboost_model import (
    FitResult,
    Progress,
    RunResult,
    build_set1_matrix,
    load_xgboost_spec,
    run_context,
)

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "predictions" / "xgboost"
RUNS_DIR = ROOT / "runs"
RUN_MD = ROOT / "kb" / "xgboost_run.md"
LOG_PATH = RUNS_DIR / "xgboost_training.log"
SUMMARY_PATH = RUNS_DIR / "xgboost_summary.txt"

RENDER_THROTTLE_S = 0.5  # min seconds between screen redraws while a fit is in progress


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------


@dataclass
class ContextState:
    name: str  # e.g. "fold_1/set1"
    status: str = "pending"  # pending | running | done | failed
    trial: int = 0
    n_trials: int = 0
    round_: int = 0
    max_round: int = 0
    result: RunResult | None = None
    error: str | None = None
    seconds: float = 0.0


@dataclass
class Dashboard(Progress):
    """Single-frame terminal redraw + append-only log, updated as training runs.

    Round/trial progress is throttled (`RENDER_THROTTLE_S`) — up to 500 rounds
    per trial would otherwise redraw the screen far more often than a human (or
    a `tmux` pane) needs to see it. The log file is never throttled: every
    context/trial boundary is written immediately, flushed, so `tail -f` from
    another terminal always shows the latest state even if the dashboard's own
    redraw is skipped.
    """

    contexts: list[ContextState]
    start: float = field(default_factory=time.monotonic)
    _current: ContextState | None = None
    _last_render: float = 0.0
    _log_fh: TextIO | None = None

    def __post_init__(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_fh = LOG_PATH.open("a", buffering=1)  # line-buffered
        self.log(f"=== run started, {len(self.contexts)} contexts ===")

    def log(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def begin_context(self, ctx: ContextState) -> None:
        ctx.status = "running"
        self._current = ctx
        self.log(f"{ctx.name}: started")
        self._render(force=True)

    def end_context(self, ctx: ContextState, result: RunResult | None, error: Exception | None) -> None:
        ctx.status = "done" if result is not None else "failed"
        ctx.result = result
        ctx.error = str(error) if error else None
        ctx.seconds = result.seconds if result else ctx.seconds
        self._current = None
        if result:
            self.log(
                f"{ctx.name}: done  rmse={result.inner_val_rmse:.5f}  "
                f"best_iter={result.best_iteration}  {result.seconds:.1f}s"
            )
        else:
            self.log(f"{ctx.name}: FAILED  {ctx.error}")
        self._render(force=True)

    # Progress interface — called from deep inside xgboost_model's search loop.
    def on_round(self, epoch: int, max_round: int) -> None:
        if self._current is not None:
            self._current.round_ = epoch
            self._current.max_round = max_round
        self._render()

    def on_trial(self, trial_idx: int, n_trials: int, result: FitResult) -> None:
        if self._current is not None:
            self._current.trial = trial_idx + 1
            self._current.n_trials = n_trials
        self.log(
            f"  trial {trial_idx + 1}/{n_trials}: rmse={result.best_rmse:.5f}  "
            f"best_iter={result.best_iteration}"
        )
        self._render(force=True)

    def _render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < RENDER_THROTTLE_S:
            return
        self._last_render = now

        width = min(shutil.get_terminal_size((100, 24)).columns, 100)
        elapsed = now - self.start
        done = [c for c in self.contexts if c.status in ("done", "failed")]
        avg = (sum(c.seconds for c in done) / len(done)) if done else None
        remaining = len(self.contexts) - len(done)
        eta = f"~{_fmt(avg * remaining)}" if avg and remaining else "?"

        lines = []
        lines.append("=" * width)
        lines.append(f"XGBoost training — #38   elapsed {_fmt(elapsed)}   eta {eta}   "
                      f"{len(done)}/{len(self.contexts)} contexts done")
        lines.append("-" * width)
        for c in self.contexts:
            mark = {"pending": " ", "running": ">", "done": "x", "failed": "!"}[c.status]
            if c.status == "running":
                extra = f"trial {c.trial}/{c.n_trials}  round {c.round_}/{c.max_round}"
            elif c.status == "done" and c.result:
                extra = f"rmse={c.result.inner_val_rmse:.5f}  {_fmt(c.result.seconds)}"
            elif c.status == "failed":
                extra = f"FAILED: {c.error}"
            else:
                extra = ""
            lines.append(f"  [{mark}] {c.name:<16s} {extra}")
        lines.append("-" * width)
        lines.append("log tail:")
        recent = LOG_PATH.read_text().splitlines()[-6:] if LOG_PATH.exists() else []
        for line in recent:
            lines.append(f"  {line[:width - 2]}")
        lines.append("=" * width)

        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        self._log_fh.close()


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _all_contexts(fs, folds) -> list[tuple[str, str]]:
    """(context, set_name) pairs in run order — also the dashboard's context list."""
    pairs = [(f"fold_{f.fold}", s) for f in folds for s in ("set1", "set2")]
    pairs += [("final", s) for s in ("set1", "set2")]
    return pairs


def run_all(dash: Dashboard) -> list[RunResult]:
    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)
    fs = load_fold_spec()
    xgb_spec = load_xgboost_spec()
    folds = fs.folds(cal)

    # Set 1's matrix is fold-invariant — build once over every origin any
    # context will ever touch (every fold's train ∪ val, plus final_train ∪
    # holdout), and slice rows out of it per context.
    parts = [fs.final_train(cal), fs.holdout(cal)]
    for f in folds:
        parts.extend([f.train, f.val])
    all_origins = parts[0].append(parts[1:]).unique().sort_values()
    dash.log("building set1 matrix (fold-invariant, built once)...")
    set1_matrix: FlatMatrix = build_set1_matrix(cal, spec, globals_, all_origins)
    dash.log(f"set1 matrix ready: {set1_matrix.values.shape}")

    final_inner_train, final_inner_val = fs.final_inner_split(cal)
    final_block = SimpleNamespace(train=fs.final_train(cal))

    specs_by_context: dict[str, dict] = {}
    for f in folds:
        specs_by_context[f"fold_{f.fold}"] = dict(
            train_origins=f.train, inner_train_origins=f.inner_train,
            inner_val_origins=f.inner_val, predict_origins=f.val, fold_for_pca=f,
        )
    specs_by_context["final"] = dict(
        train_origins=fs.final_train(cal), inner_train_origins=final_inner_train,
        inner_val_origins=final_inner_val, predict_origins=fs.holdout(cal), fold_for_pca=final_block,
    )

    results: list[RunResult] = []
    ctx_by_name = {c.name: c for c in dash.contexts}
    for context, set_name in _all_contexts(fs, folds):
        ctx = ctx_by_name[f"{context}/{set_name}"]
        dash.begin_context(ctx)
        try:
            kwargs = specs_by_context[context]
            r = run_context(
                cal, spec, globals_, xgb_spec,
                context=context, set_name=set_name,
                set1_matrix=set1_matrix if set_name == "set1" else None,
                progress=dash,
                **kwargs,
            )
            results.append(r)
            dash.end_context(ctx, r, None)
        except Exception as exc:  # noqa: BLE001 - one bad context must not kill the run
            dash.end_context(ctx, None, exc)

    return results


def persist(results: list[RunResult], n_trials: int) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = PRED_DIR / f"{r.context}_{r.set_name}.parquet"
        r.predictions.to_parquet(path)

    lines = [
        "# XGBoost run — predictions and winning hyperparameters",
        "",
        "Generated by `scripts/train_xgboost.py`. Do not hand-edit — rerun the script.",
        "",
        "Resolves [#38](https://github.com/jenujari/prdict-pov-v1/issues/38)'s "
        "\"predictions are emitted per origin, and the run is reproducible\". "
        "Search space and budget are fixed in `kb/xgboost_spec.md`, written before this ran.",
        "",
        "| Context | Set | Train rows | Predict rows | Winning params | Best iter | Inner rmse | Seconds |",
        "|---------|-----|-----------|--------------|-----------------|-----------|------------|---------|",
    ]
    total_seconds = 0.0
    for r in results:
        total_seconds += r.seconds
        params = ", ".join(f"{k}={v}" for k, v in r.winning_params.items())
        lines.append(
            f"| {r.context} | {r.set_name} | {r.n_train} | {r.n_predict} | "
            f"{params} | {r.best_iteration} | {r.inner_val_rmse:.5f} | {r.seconds:.1f} |"
        )
    lines.append("")
    lines.append(f"Total wall clock: {total_seconds:.1f}s across {len(results)} contexts "
                  f"({n_trials} search trials + 1 refit each).")
    lines.append("")
    lines.append("Predictions in `predictions/xgboost/*.parquet` — origin date index, "
                  "`y_pred_1..y_pred_10`.")
    RUN_MD.write_text("\n".join(lines) + "\n")


def write_summary(dash: Dashboard, elapsed: float) -> None:
    """A plain-text summary meant to be pasted back into a Claude Code session."""
    done = [c for c in dash.contexts if c.status == "done"]
    failed = [c for c in dash.contexts if c.status == "failed"]
    pending = [c for c in dash.contexts if c.status not in ("done", "failed")]

    lines = [
        "XGBoost training run — summary",
        f"Finished: {datetime.now().isoformat(timespec='seconds')}",
        f"Total wall clock: {_fmt(elapsed)}",
        f"Contexts: {len(done)} done, {len(failed)} failed, {len(pending)} not reached",
        "",
    ]
    if done:
        lines.append("Completed:")
        for c in sorted(done, key=lambda c: c.name):
            r = c.result
            lines.append(
                f"  {c.name:<16s} rmse={r.inner_val_rmse:.5f}  best_iter={r.best_iteration}  "
                f"{_fmt(r.seconds)}  params={r.winning_params}"
            )
        lines.append("")
    if failed:
        lines.append("Failed:")
        for c in sorted(failed, key=lambda c: c.name):
            lines.append(f"  {c.name:<16s} {c.error}")
        lines.append("")
    if pending:
        lines.append("Not reached (run stopped early): " + ", ".join(c.name for c in pending))
        lines.append("")
    lines.append(f"Predictions: predictions/xgboost/*.parquet ({len(done)} files)")
    lines.append("Run table: kb/xgboost_run.md")
    lines.append("Full log: runs/xgboost_training.log")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main() -> None:
    fs = load_fold_spec()
    cal = load_calendar()
    folds = fs.folds(cal)
    contexts = [ContextState(name=f"{c}/{s}") for c, s in _all_contexts(fs, folds)]
    dash = Dashboard(contexts=contexts)

    start = time.monotonic()
    try:
        results = run_all(dash)
    except KeyboardInterrupt:
        dash.log("=== interrupted by user ===")
        results = [c.result for c in dash.contexts if c.result is not None]
    finally:
        elapsed = time.monotonic() - start
        n_trials = load_xgboost_spec()["search"]["n_trials"]
        if results:
            persist(results, n_trials=n_trials)
        write_summary(dash, elapsed)
        dash.close()


if __name__ == "__main__":
    main()
