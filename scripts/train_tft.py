"""Train the TFT on both input sets, across every fold and the final holdout refit.

Resolves wayfinder ticket #39. Reads the fixed training contract from
`kb/tft_spec.json` (`scripts/build_tft_spec.py`, committed separately and
first) and `prdict.tft_model`'s dataset builders and search logic. This script
is only the orchestration — mirrors `scripts/train_xgboost.py`'s shape exactly
(same Dashboard/log/checkpoint pattern), adapted from XGBoost's round/trial
granularity to TFT's epoch/trial granularity.

12 contexts — 5 folds x {set1, set2} (search + refit on `fold.train`, predict
on `fold.val`) plus the final holdout arm x {set1, set2} (search + refit on
`final_train`, predict on `holdout`) — each running the fixed 4-trial search
from `kb/tft_spec.json`, 60 total TFT fits. A context that raises is logged
and skipped rather than aborting the whole run.

**Resumable per context**, same granularity as #38: each context's prediction
parquet is written the moment it finishes, alongside a metadata sidecar
recording which `kb/tft_spec.json` it ran under. Relaunching this script skips
every context whose sidecar matches the *current* spec. Unlike XGBoost's
`xgb.train()`, a TFT fit does have a natural mid-fit resume point (Lightning
checkpoints), but a context interrupted mid-fit still restarts its search+
refit from scratch here — wiring genuine sub-context resume would need
per-trial checkpointing this ticket doesn't build, so the honest granularity
is: a context is the unit of resume, exactly like #38.

Output:
  predictions/tft/{fold1..5,final}_{set1,set2}.parquet
      origin date index, y_pred_1..y_pred_10 — same schema as #38's XGBoost
      output, so #14's scorecard needs zero changes to score these.
  predictions/tft/{fold1..5,final}_{set1,set2}.meta.json
      the resume checkpoint.
  kb/tft_run.md
      winning hyperparameters, inner-val loss, and timing per context.
  runs/tft_training.log / runs/tft_summary.txt
      same role as #38's xgboost_training.log/xgboost_summary.txt.

Run via ./scripts/run_tft_training.sh, or directly:
    uv run python scripts/train_tft.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

import pandas as pd

from prdict import encoding
from prdict.folds import load_fold_spec
from prdict.trading_calendar import load_calendar
from prdict.tft_model import FitResult, Progress, RunResult, load_tft_spec, run_context

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "predictions" / "tft"
RUNS_DIR = ROOT / "runs"
RUN_MD = ROOT / "kb" / "tft_run.md"
LOG_PATH = RUNS_DIR / "tft_training.log"
SUMMARY_PATH = RUNS_DIR / "tft_summary.txt"
CKPT_DIR = RUNS_DIR / "tft_checkpoints"

RENDER_THROTTLE_S = 0.5


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------


@dataclass
class ContextState:
    name: str  # e.g. "fold_1/set1"
    status: str = "pending"  # pending | running | done | failed
    trial: int = 0
    n_trials: int = 0
    epoch: int = 0
    max_epoch: int = 0
    result: RunResult | None = None
    error: str | None = None
    seconds: float = 0.0


@dataclass
class Dashboard(Progress):
    """Single-frame terminal redraw + append-only log — same shape as #38's."""

    contexts: list[ContextState]
    start: float = field(default_factory=time.monotonic)
    _current: ContextState | None = None
    _last_render: float = 0.0
    _log_fh: TextIO | None = None

    def __post_init__(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_fh = LOG_PATH.open("a", buffering=1)
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
                f"{ctx.name}: done  val_loss={result.inner_val_loss:.5f}  "
                f"best_epoch={result.best_epoch}  {result.seconds:.1f}s"
            )
        else:
            self.log(f"{ctx.name}: FAILED  {ctx.error}")
        self._render(force=True)

    def mark_resumed(self, ctx: ContextState, result: RunResult) -> None:
        ctx.status, ctx.result, ctx.seconds = "done", result, result.seconds
        self.log(f"{ctx.name}: resumed from checkpoint  val_loss={result.inner_val_loss:.5f}")
        self._render(force=True)

    # Progress interface — called from deep inside tft_model's search loop.
    def on_epoch(self, epoch: int, max_epochs: int) -> None:
        if self._current is not None:
            self._current.epoch = epoch
            self._current.max_epoch = max_epochs
        self._render()

    def on_trial(self, trial_idx: int, n_trials: int, result: FitResult) -> None:
        if self._current is not None:
            self._current.trial = trial_idx + 1
            self._current.n_trials = n_trials
        self.log(
            f"  trial {trial_idx + 1}/{n_trials}: val_loss={result.best_val_loss:.5f}  "
            f"best_epoch={result.best_epoch}"
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
        lines.append(f"TFT training — #39   elapsed {_fmt(elapsed)}   eta {eta}   "
                      f"{len(done)}/{len(self.contexts)} contexts done")
        lines.append("-" * width)
        for c in self.contexts:
            mark = {"pending": " ", "running": ">", "done": "x", "failed": "!"}[c.status]
            if c.status == "running":
                extra = f"trial {c.trial}/{c.n_trials}  epoch {c.epoch}/{c.max_epoch}"
            elif c.status == "done" and c.result:
                extra = f"val_loss={c.result.inner_val_loss:.5f}  {_fmt(c.result.seconds)}"
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


def _all_contexts(folds) -> list[tuple[str, str]]:
    pairs = [(f"fold_{f.fold}", s) for f in folds for s in ("set1", "set2")]
    pairs += [("final", s) for s in ("set1", "set2")]
    return pairs


# --------------------------------------------------------------------------
# per-context checkpointing — what makes the run resumable
# --------------------------------------------------------------------------


def spec_fingerprint(tft_spec: dict) -> str:
    return hashlib.sha256(json.dumps(tft_spec, sort_keys=True).encode()).hexdigest()[:16]


def _parquet_path(context: str, set_name: str) -> Path:
    return PRED_DIR / f"{context}_{set_name}.parquet"


def _meta_path(context: str, set_name: str) -> Path:
    return PRED_DIR / f"{context}_{set_name}.meta.json"


def _persist_context(result: RunResult, fingerprint: str) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(_parquet_path(result.context, result.set_name))
    meta = {
        "spec_fingerprint": fingerprint,
        "winning_params": result.winning_params,
        "best_epoch": result.best_epoch,
        "inner_val_loss": result.inner_val_loss,
        "n_train": result.n_train,
        "n_predict": result.n_predict,
        "seconds": result.seconds,
    }
    _meta_path(result.context, result.set_name).write_text(json.dumps(meta, indent=2))


def _load_checkpoint(context: str, set_name: str, fingerprint: str) -> RunResult | None:
    ppath, mpath = _parquet_path(context, set_name), _meta_path(context, set_name)
    if not (ppath.exists() and mpath.exists()):
        return None
    meta = json.loads(mpath.read_text())
    if meta["spec_fingerprint"] != fingerprint:
        return None
    return RunResult(
        context=context,
        set_name=set_name,
        predictions=pd.read_parquet(ppath),
        winning_params=meta["winning_params"],
        best_epoch=meta["best_epoch"],
        inner_val_loss=meta["inner_val_loss"],
        n_train=meta["n_train"],
        n_predict=meta["n_predict"],
        seconds=meta["seconds"],
    )


def run_all(dash: Dashboard) -> list[RunResult]:
    cal = load_calendar()
    spec = encoding.load_spec()
    globals_ = encoding.load_global(spec)
    fs = load_fold_spec()
    tft_spec = load_tft_spec()
    folds = fs.folds(cal)

    final_inner_train, final_inner_val = fs.final_inner_split(cal)
    final_block = SimpleNamespace(train=fs.final_train(cal))

    specs_by_context: dict[str, dict] = {}
    for f in folds:
        specs_by_context[f"fold_{f.fold}"] = dict(
            train_origins=f.train, inner_train_origins=f.inner_train,
            inner_val_origins=f.inner_val, predict_origins=f.val, fold_or_final=f,
        )
    specs_by_context["final"] = dict(
        train_origins=fs.final_train(cal), inner_train_origins=final_inner_train,
        inner_val_origins=final_inner_val, predict_origins=fs.holdout(cal), fold_or_final=final_block,
    )

    fingerprint = spec_fingerprint(tft_spec)
    results: list[RunResult] = []
    ctx_by_name = {c.name: c for c in dash.contexts}
    for context, set_name in _all_contexts(folds):
        ctx = ctx_by_name[f"{context}/{set_name}"]

        checkpoint = _load_checkpoint(context, set_name, fingerprint)
        if checkpoint is not None:
            results.append(checkpoint)
            dash.mark_resumed(ctx, checkpoint)
            continue

        dash.begin_context(ctx)
        try:
            kwargs = specs_by_context[context]
            ckpt_dir = CKPT_DIR / f"{context}_{set_name}"
            r = run_context(
                cal, spec, globals_, tft_spec,
                context=context, set_name=set_name,
                checkpoint_dir=ckpt_dir,
                progress=dash,
                **kwargs,
            )
            _persist_context(r, fingerprint)
            results.append(r)
            dash.end_context(ctx, r, None)
        except Exception as exc:  # noqa: BLE001 - one bad context must not kill the run
            dash.end_context(ctx, None, exc)

        write_run_table(results, n_trials=tft_spec["search"]["n_trials"])
        write_summary(dash, time.monotonic() - dash.start)

    return results


def write_run_table(results: list[RunResult], n_trials: int) -> None:
    lines = [
        "# TFT run — predictions and winning hyperparameters",
        "",
        "Generated by `scripts/train_tft.py`. Do not hand-edit — rerun the script.",
        "",
        "Resolves [#39](https://github.com/jenujari/prdict-pov-v1/issues/39)'s "
        "\"predictions are emitted per origin, and the run is reproducible\". "
        "Search space and budget are fixed in `kb/tft_spec.md`, written before this ran.",
        "",
        "| Context | Set | Train rows | Predict rows | Winning params | Best epoch | Inner val loss | Seconds |",
        "|---------|-----|-----------|--------------|-----------------|-----------|------------|---------|",
    ]
    total_seconds = 0.0
    for r in results:
        total_seconds += r.seconds
        params = ", ".join(f"{k}={v}" for k, v in r.winning_params.items())
        lines.append(
            f"| {r.context} | {r.set_name} | {r.n_train} | {r.n_predict} | "
            f"{params} | {r.best_epoch} | {r.inner_val_loss:.5f} | {r.seconds:.1f} |"
        )
    lines.append("")
    lines.append(f"Total wall clock: {total_seconds:.1f}s across {len(results)} contexts "
                  f"({n_trials} search trials + 1 refit each).")
    lines.append("")
    lines.append("Predictions in `predictions/tft/*.parquet` — origin date index, "
                  "`y_pred_1..y_pred_10` (same schema as #38's XGBoost output).")
    RUN_MD.write_text("\n".join(lines) + "\n")


def write_summary(dash: Dashboard, elapsed: float) -> None:
    done = [c for c in dash.contexts if c.status == "done"]
    failed = [c for c in dash.contexts if c.status == "failed"]
    pending = [c for c in dash.contexts if c.status not in ("done", "failed")]

    lines = [
        "TFT training run — summary",
        f"Last updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Elapsed so far: {_fmt(elapsed)}",
        f"Contexts: {len(done)} done, {len(failed)} failed, {len(pending)} not reached yet",
        "",
    ]
    if done:
        lines.append("Completed:")
        for c in sorted(done, key=lambda c: c.name):
            r = c.result
            lines.append(
                f"  {c.name:<16s} val_loss={r.inner_val_loss:.5f}  best_epoch={r.best_epoch}  "
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
    lines.append(f"Predictions: predictions/tft/*.parquet ({len(done)} files)")
    lines.append("Run table: kb/tft_run.md")
    lines.append("Full log: runs/tft_training.log")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main() -> None:
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    fs = load_fold_spec()
    cal = load_calendar()
    folds = fs.folds(cal)
    contexts = [ContextState(name=f"{c}/{s}") for c, s in _all_contexts(folds)]
    dash = Dashboard(contexts=contexts)

    results: list[RunResult] = [c.result for c in dash.contexts if c.result is not None]
    try:
        results = run_all(dash)
    except (KeyboardInterrupt, SIGTERMException):
        dash.log("=== interrupted — already-completed contexts are checkpointed ===")
        results = [c.result for c in dash.contexts if c.result is not None]
    finally:
        n_trials = load_tft_spec()["search"]["n_trials"]
        if results:
            write_run_table(results, n_trials=n_trials)
        write_summary(dash, time.monotonic() - dash.start)
        dash.close()


if __name__ == "__main__":
    main()
