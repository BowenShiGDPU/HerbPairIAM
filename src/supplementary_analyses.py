"""Backward-compat shim. Real code lives in ``src/evaluation/supplementary_analyses.py``."""
from evaluation.supplementary_analyses import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.supplementary_analyses import main as _main

    raise SystemExit(_main())
