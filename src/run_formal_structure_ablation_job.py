"""Backward-compat shim. Real code lives in ``src/scripts/run_formal_structure_ablation_job.py``."""
from scripts.run_formal_structure_ablation_job import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.run_formal_structure_ablation_job import main as _main

    raise SystemExit(_main())
