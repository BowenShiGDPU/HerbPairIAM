"""Backward-compat shim. Real code lives in ``src/data/phase2_dataset.py``."""
from data.phase2_dataset import *  # noqa: F401,F403

if __name__ == "__main__":
    from data.phase2_dataset import build_dataset

    build_dataset()
