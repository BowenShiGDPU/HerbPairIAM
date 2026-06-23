"""Backward-compat shim. Real code lives in ``src/evaluation/compute_profile_postprocess.py``."""
from evaluation.compute_profile_postprocess import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.compute_profile_postprocess import main as _main

    raise SystemExit(_main())
