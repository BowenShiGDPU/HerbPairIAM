"""Backward-compat shim. Real code lives in ``src/evaluation/finalize_results.py``."""
from evaluation.finalize_results import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.finalize_results import main as _main

    _main()
