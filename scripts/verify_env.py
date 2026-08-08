"""Smoke-test the native runtime.

Confirms the ML stack imports and that torch and xgboost actually run, not just
import. On the glibc host this runs directly:

    uv run python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "pyarrow",
    "matplotlib",
    "xgboost",
    "torch",
    "pytorch_forecasting",
    "lightning",
]


def main() -> int:
    print(f"python {sys.version.split()[0]}  libc={platform.libc_ver()}")
    print(f"prefix {sys.prefix}")

    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:22s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 - report every failure, not the first
            print(f"  {name:22s} FAIL {type(exc).__name__}: {exc}")
            failed.append(name)

    if failed:
        print(f"\nFAILED: {failed}")
        return 1

    import torch
    import xgboost as xgb

    print(f"\ntorch cuda={torch.version.cuda} threads={torch.get_num_threads()}")

    # Exercise, do not merely import.
    x = torch.randn(64, 8)
    print(f"torch matmul ok: {(x @ x.T).shape}")

    import numpy as np

    booster = xgb.train(
        {"tree_method": "hist", "multi_strategy": "multi_output_tree"},
        xgb.DMatrix(np.random.randn(128, 5), label=np.random.randn(128, 10)),
        num_boost_round=2,
    )
    shape = booster.predict(xgb.DMatrix(np.random.randn(4, 5))).shape
    print(f"xgboost multi_output_tree ok: predict shape {shape}")

    repo = Path(__file__).resolve().parent.parent
    print(f"\nrepo: {repo}  nft50.csv: {(repo / 'nft50.csv').exists()}")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
