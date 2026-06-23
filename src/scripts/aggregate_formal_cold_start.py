"""Aggregate cold-start cell pickles into the formal table.

Reads ``<stage>/fold_results/cold_<split_type>_<seed>_<model>.pkl`` and writes
both ``tables/cold_start.csv`` (one row per (split_type, model)) and
``supplementary/cold_start_progress.csv`` (one row per (split_type, seed,
model)).

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/cold_start \\
    python -u src/aggregate_formal_cold_start.py
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import (
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    load_pickle,
    sanitize_name,
)
from phase4_evaluation import (
    COLD_START_GRAPH_MODELS,
    COLD_START_NEURAL_MODELS,
    COLD_START_TABULAR_MODELS,
    PRIMARY_MODEL_NAME,
)


sys.stdout.reconfigure(line_buffering=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_dirs()
    if not FOLD_RESULTS_DIR.exists():
        print(f"Fold-results dir not found: {FOLD_RESULTS_DIR}")
        return 1
    pattern = re.compile(r"^cold_(?P<split>[a-zA-Z]+)_(?P<seed>\d+)_(?P<name>.+)\.pkl$")
    sanitized_to_model: dict[str, str] = {}
    candidates = sorted({*COLD_START_TABULAR_MODELS, *COLD_START_NEURAL_MODELS, *COLD_START_GRAPH_MODELS, PRIMARY_MODEL_NAME})
    for model in candidates:
        sanitized_to_model[sanitize_name(model)] = model

    rows = []
    for path in sorted(FOLD_RESULTS_DIR.glob("cold_*_fold*.pkl")) + sorted(FOLD_RESULTS_DIR.glob("cold_*.pkl")):
        m = pattern.match(path.name)
        if not m:
            continue
        split_type = m.group("split").capitalize()
        seed = int(m.group("seed"))
        sname = m.group("name")
        model = sanitized_to_model.get(sname, sname)
        try:
            obj = load_pickle(path)
        except Exception:
            continue
        rows.append({
            "split_type": split_type,
            "seed": seed,
            "model": model,
            "auroc": float(obj.get("auroc", float("nan"))),
            "auprc": float(obj.get("auprc", float("nan"))),
            "precision": float(obj.get("precision", float("nan"))),
            "recall": float(obj.get("recall", float("nan"))),
            "f1": float(obj.get("f1", float("nan"))),
            "mcc": float(obj.get("mcc", float("nan"))),
            "n_test": int(obj.get("n_test", len(obj.get("y_true", [])))),
        })
    if not rows:
        print(f"No cold_*.pkl files in {FOLD_RESULTS_DIR}")
        return 1
    progress = pd.DataFrame(rows).sort_values(["split_type", "model", "seed"]).reset_index(drop=True)
    progress.to_csv(SUPP_DIR / "cold_start_progress.csv", index=False)

    by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    for (split_type, model), grp in progress.groupby(["split_type", "model"]):
        by_pair[(split_type, model)] = grp

    agg_rows = []
    for (split_type, model), grp in by_pair.items():
        agg_rows.append({
            "Model": model,
            "split_type": split_type,
            "n_seeds": int(len(grp)),
            "AUROC_mean": float(grp["auroc"].mean()),
            "AUROC_std": float(grp["auroc"].std(ddof=0)),
            "AUPRC_mean": float(grp["auprc"].mean()),
            "AUPRC_std": float(grp["auprc"].std(ddof=0)),
            "F1_mean": float(grp["f1"].mean()),
            "MCC_mean": float(grp["mcc"].mean()),
        })
    out_df = pd.DataFrame(agg_rows).sort_values(["split_type", "Model"]).reset_index(drop=True)
    out_df.to_csv(TABLES_DIR / "cold_start.csv", index=False)
    print(f"Wrote {TABLES_DIR / 'cold_start.csv'} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
