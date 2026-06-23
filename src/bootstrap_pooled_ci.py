"""Backward-compat shim. Real code lives in ``src/evaluation/bootstrap_pooled_ci.py``."""
from evaluation.bootstrap_pooled_ci import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.bootstrap_pooled_ci import main as _main

    raise SystemExit(_main())
