"""Backward-compat shim. Real code lives in ``src/scripts/run_doseaware_sensitivity.py``."""
from scripts.run_doseaware_sensitivity import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.run_doseaware_sensitivity import main as _main

    _main()
