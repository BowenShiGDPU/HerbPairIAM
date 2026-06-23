"""Executable entrypoints for the formal benchmark.

Each ``run_formal_*_job.py`` runs one (model, fold/seed/ratio) cell with auto-skip;
each ``aggregate_formal_*.py`` rolls fold pickles into the final supplementary table.
``run_formal_stage.py`` is the single dispatcher used by the formal pipeline.

Note: every script in this directory inserts ``../`` (the ``src/`` root) into
``sys.path`` at startup so the historical bare-name imports such as
``from neural_models import ...`` continue to resolve through the root-level
shim files (``src/neural_models.py``).
"""
