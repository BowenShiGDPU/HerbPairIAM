"""Backward-compat shim. Real code lives in ``src/scripts/aggregate_formal_cold_start.py``."""
from scripts.aggregate_formal_cold_start import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.aggregate_formal_cold_start import main as _main

    raise SystemExit(_main())
