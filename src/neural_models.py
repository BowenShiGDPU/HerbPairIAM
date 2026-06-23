"""Backward-compat shim. Real code lives in ``src/models/neural_models.py``.

Imports here exist so historical bare-name imports such as
``from neural_models import ModelConfig`` keep working when ``src/`` is on
``sys.path``. New code should import from ``models.neural_models`` directly.
"""
from models.neural_models import *  # noqa: F401,F403
