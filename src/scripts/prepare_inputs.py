"""Load dataset + KG features and optionally write a run manifest.

This is the lightweight dependency used by the training scripts in this
directory. It is the only function from the original evaluation module
that the open-source entry points need.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import numpy as np

from experiment_utils import ensure_output_dirs, write_run_manifest
from neural_models import load_all


PRIMARY_MODEL_NAME = "HerbPairIAM"


def prepare_common_inputs():
    """Return ``(ds, df, feature_cols, X, labels, hp, ap, pf, lookups)``.

    ``ds`` is the full dataset pickle built by ``data.phase2_dataset``,
    ``df`` the pair table, ``X`` the dense feature matrix for tabular
    baselines, and ``hp / ap / pf / lookups`` the KG artefacts built by
    ``data.phase1_precompute`` (needed by the neural models).
    """

    ensure_output_dirs()
    try:
        write_run_manifest()
    except Exception as exc:
        print(f"[warn] write_run_manifest failed: {exc}", flush=True)
    ds, hp, ap, pf, lookups = load_all()
    df = ds["df"]
    labels = df["label"].values.astype(int)
    feature_cols = ds["feature_cols"]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32), 0.0)
    return ds, df, feature_cols, X, labels, hp, ap, pf, lookups
