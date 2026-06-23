"""Backward-compat shim. Real code lives in ``src/evaluation/make_figures.py``."""
from evaluation.make_figures import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.make_figures import main as _main

    raise SystemExit(_main())
