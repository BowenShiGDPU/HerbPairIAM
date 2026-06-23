"""Backward-compat shim. Real code lives in ``src/evaluation/phase5_interpretability.py``."""
from evaluation.phase5_interpretability import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.phase5_interpretability import main as _main

    raise SystemExit(_main())
