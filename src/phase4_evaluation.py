"""Backward-compat shim. Real code lives in ``src/evaluation/phase4_evaluation.py``.

Note: ``from x import *`` skips underscore-prefixed names, so we explicitly
re-export the private resumable helpers that scripts under ``src/scripts/``
import by name.
"""
from evaluation.phase4_evaluation import *  # noqa: F401,F403
from evaluation.phase4_evaluation import (  # noqa: F401
    _resumable_neural_cv,
    _resumable_tabular_cv,
    _resumable_neg_graph_cv,
    _resumable_neg_neural_cv,
    _resumable_neg_tabular_cv,
    _resumable_cold_graph,
    _resumable_cold_neural,
    _resumable_cold_tabular,
)

if __name__ == "__main__":
    from evaluation.phase4_evaluation import main as _main

    _main()
