"""Backward-compat shim. Real code lives in ``src/scripts/aggregate_formal_structure_ablation.py``."""
from scripts.aggregate_formal_structure_ablation import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.aggregate_formal_structure_ablation import main as _main

    raise SystemExit(_main())
