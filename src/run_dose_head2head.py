"""Backward-compat shim. Real code lives in ``src/scripts/run_dose_head2head.py``."""
from scripts.run_dose_head2head import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.run_dose_head2head import main as _main

    raise SystemExit(_main())
