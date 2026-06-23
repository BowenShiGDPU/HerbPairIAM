"""Backward-compat shim. Real code lives in ``src/scripts/run_formal_stage.py``."""
from scripts.run_formal_stage import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.run_formal_stage import main as _main

    _main()
