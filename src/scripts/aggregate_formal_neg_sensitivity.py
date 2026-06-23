"""Aggregate per (model, neg_ratio) pickles into the supplementary table.

Reads ``<stage>/fold_results/<model>_neg<R>_fold<K>.pkl`` and writes
``<stage>/supplementary/neg_sampling_sensitivity.csv`` with the columns
required by EXPERIMENT_PROTOCOL.md s9.1, plus a paired t-test against
DoseAwareIAM @ 1:10 to make the headline argument "neg-ratio choice does not
change the conclusion".

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/neg_sampling_sensitivity \\
    python -u src/aggregate_formal_neg_sensitivity.py
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
from scipy import stats

from experiment_utils import (
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    ensure_output_dirs,
    load_pickle,
    sanitize_name,
)
from phase4_evaluation import (
    NEG_RATIO_GRID,
    NEG_SENSITIVITY_GRAPH,
    NEG_SENSITIVITY_NEURAL,
    NEG_SENSITIVITY_PRIMARY,
    NEG_SENSITIVITY_TABULAR,
)


# The tested-universe negative pool gives each training fold a finite n_neg.
# balance_train_indices() samples min(n_neg, n_pos * neg_ratio) negatives
# without replacement; once neg_ratio * n_pos >= n_neg the sampler saturates
# and larger ratios train the same model. For our 707-label dataset the
# n_neg/n_pos ratio per training fold is ~4.63, so every ratio >= 5 is
# numerically indistinguishable from the ratio-5 run. We flag those rows in
# the sensitivity CSV so readers/reviewers are not misled.
NEGATIVE_POOL_SATURATION_RATIO = 5


sys.stdout.reconfigure(line_buffering=True)


def _summary(results: list[dict]) -> dict:
    out = {}
    for key in ["auroc", "auprc", "precision", "recall", "f1", "mcc"]:
        vals = np.asarray([float(r[key]) for r in results], dtype=float)
        out[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{key}_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
    return out


def _model_kind(model: str) -> str:
    if model in NEG_SENSITIVITY_TABULAR:
        return "tabular"
    if model in NEG_SENSITIVITY_NEURAL:
        return "neural"
    if model in NEG_SENSITIVITY_GRAPH:
        return "graph"
    return "other"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_dirs()
    if not FOLD_RESULTS_DIR.exists():
        print(f"No fold pickles in {FOLD_RESULTS_DIR}")
        return 1
    pattern = re.compile(r"^(?P<name>.+)_neg(?P<ratio>\d+)_fold(?P<fold>\d+)\.pkl$")
    by_model_ratio: dict[tuple[str, int], list[dict]] = {}
    sanitized_to_model: dict[str, str] = {}
    candidate_models = sorted({*NEG_SENSITIVITY_TABULAR, *NEG_SENSITIVITY_NEURAL, *NEG_SENSITIVITY_GRAPH, NEG_SENSITIVITY_PRIMARY})
    for model in candidate_models:
        sanitized_to_model[sanitize_name(model)] = model
    for path in sorted(FOLD_RESULTS_DIR.glob("*_neg*_fold*.pkl")):
        m = pattern.match(path.name)
        if not m:
            continue
        sname = m.group("name")
        model = sanitized_to_model.get(sname, sname)
        ratio = int(m.group("ratio"))
        try:
            obj = load_pickle(path)
        except Exception:
            continue
        by_model_ratio.setdefault((model, ratio), []).append(obj)
    if not by_model_ratio:
        print(f"Did not find any *_neg*_fold*.pkl files under {FOLD_RESULTS_DIR}")
        return 1

    rows = []
    primary_pivot = sorted(
        by_model_ratio.get((NEG_SENSITIVITY_PRIMARY, 10), []),
        key=lambda r: int(r.get("fold", 0)),
    )
    primary_auroc = (
        np.asarray([float(r["auroc"]) for r in primary_pivot], dtype=float)
        if primary_pivot
        else None
    )
    primary_auprc = (
        np.asarray([float(r["auprc"]) for r in primary_pivot], dtype=float)
        if primary_pivot
        else None
    )

    for (model, ratio), results in sorted(by_model_ratio.items()):
        results = sorted(results, key=lambda r: int(r.get("fold", 0)))
        summary = _summary(results)
        row = {
            "model": model,
            "type": _model_kind(model),
            "neg_ratio": int(ratio),
            "n_folds": int(len(results)),
            "auroc_mean": summary["auroc_mean"],
            "auroc_std": summary["auroc_std"],
            "auprc_mean": summary["auprc_mean"],
            "auprc_std": summary["auprc_std"],
            "precision_mean": summary["precision_mean"],
            "recall_mean": summary["recall_mean"],
            "f1_mean": summary["f1_mean"],
            "mcc_mean": summary["mcc_mean"],
            # Flag rows where the negative pool is exhausted and the ratio
            # choice becomes numerically redundant. See the module-level
            # NEGATIVE_POOL_SATURATION_RATIO constant and the comment there.
            "saturated": bool(int(ratio) > NEGATIVE_POOL_SATURATION_RATIO),
        }
        if primary_auroc is not None and len(primary_pivot) == len(results) and len(results) >= 2:
            sib_auroc = np.asarray([float(r["auroc"]) for r in results], dtype=float)
            sib_auprc = np.asarray([float(r["auprc"]) for r in results], dtype=float)
            try:
                _, p1 = stats.ttest_rel(primary_auroc, sib_auroc)
            except Exception:
                p1 = float("nan")
            try:
                _, p2 = stats.ttest_rel(primary_auprc, sib_auprc)
            except Exception:
                p2 = float("nan")
            row["p_AUROC_vs_primary_neg10"] = float(p1)
            row["p_AUPRC_vs_primary_neg10"] = float(p2)
        else:
            row["p_AUROC_vs_primary_neg10"] = float("nan")
            row["p_AUPRC_vs_primary_neg10"] = float("nan")
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["model", "neg_ratio"]).reset_index(drop=True)
    out_path = SUPP_DIR / "neg_sampling_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
