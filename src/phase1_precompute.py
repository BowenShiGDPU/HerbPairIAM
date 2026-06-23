"""Backward-compat shim. Real code lives in ``src/data/phase1_precompute.py``."""
from data.phase1_precompute import *  # noqa: F401,F403

if __name__ == "__main__":
    from data.phase1_precompute import main as _main

    _main()
