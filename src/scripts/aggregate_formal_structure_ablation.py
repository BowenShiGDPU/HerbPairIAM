"""Aggregate structure-ablation fold pickles into ``structure_ablation.csv``.

Produces the table called for in EXPERIMENT_PROTOCOL.md s6.4 with paired
t-test against ``DoseAwareIAM`` (the primary structure) and
Holm-Bonferroni-adjusted p-values.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/structure_ablation \\
    python -u src/aggregate_formal_structure_ablation.py
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import (
    FOLD_RESULTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    holm_bonferroni,
    load_pickle,
)
from phase4_evaluation import DOSEAWARE_STRUCTURE_MODELS, PRIMARY_MODEL_NAME
from neural_models import summarize_results as summarize_neural


sys.stdout.reconfigure(line_buffering=True)


def _load_tag_results(tag: str) -> list[dict]:
    results = []
    for fold_id in range(10):
        path = FOLD_RESULTS_DIR / f"{tag}_fold{fold_id}.pkl"
        if not path.exists():
            continue
        try:
            results.append(load_pickle(path))
        except Exception:
            continue
    return sorted(results, key=lambda item: int(item.get("fold", 0)))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_dirs()
    rows = []
    fold_results: dict[str, list[dict]] = {}
    for model in DOSEAWARE_STRUCTURE_MODELS:
        results = _load_tag_results(model)
        if not results:
            continue
        fold_results[model] = results
        summary = summarize_neural(results)
        rows.append({
            "Model": model,
            "n_folds": int(len(results)),
            "AUROC": float(summary["auroc_mean"]),
            "AUPRC": float(summary["auprc_mean"]),
            "AUROC_std": float(summary["auroc_std"]),
            "AUPRC_std": float(summary["auprc_std"]),
        })

    if not rows:
        print(f"No structure-ablation fold pickles found in {FOLD_RESULTS_DIR}")
        return 1

    primary_rows = [r for r in rows if r["Model"] == PRIMARY_MODEL_NAME]
    if primary_rows:
        primary_summary = primary_rows[0]
        primary_auroc = primary_summary["AUROC"]
        primary_auprc = primary_summary["AUPRC"]
        if PRIMARY_MODEL_NAME in fold_results:
            primary_per_fold_auroc = np.asarray([float(r["auroc"]) for r in fold_results[PRIMARY_MODEL_NAME]], dtype=float)
            primary_per_fold_auprc = np.asarray([float(r["auprc"]) for r in fold_results[PRIMARY_MODEL_NAME]], dtype=float)
        else:
            primary_per_fold_auroc = primary_per_fold_auprc = None
    else:
        primary_summary = None
        primary_per_fold_auroc = primary_per_fold_auprc = None

    pvals_auroc: dict[str, float] = {}
    pvals_auprc: dict[str, float] = {}
    for r in rows:
        model = r["Model"]
        if primary_summary is None:
            r["delta_AUROC_vs_primary"] = float("nan")
            r["delta_AUPRC_vs_primary"] = float("nan")
            r["p_AUROC_vs_primary"] = float("nan")
            r["p_AUPRC_vs_primary"] = float("nan")
            continue
        r["delta_AUROC_vs_primary"] = float(r["AUROC"] - primary_summary["AUROC"])
        r["delta_AUPRC_vs_primary"] = float(r["AUPRC"] - primary_summary["AUPRC"])
        if model == PRIMARY_MODEL_NAME or primary_per_fold_auroc is None:
            r["p_AUROC_vs_primary"] = float("nan")
            r["p_AUPRC_vs_primary"] = float("nan")
            continue
        sibling_auroc = np.asarray([float(x["auroc"]) for x in fold_results[model]], dtype=float)
        sibling_auprc = np.asarray([float(x["auprc"]) for x in fold_results[model]], dtype=float)
        if sibling_auroc.size != primary_per_fold_auroc.size:
            r["p_AUROC_vs_primary"] = float("nan")
            r["p_AUPRC_vs_primary"] = float("nan")
            continue
        try:
            _, p1 = stats.ttest_rel(primary_per_fold_auroc, sibling_auroc)
        except Exception:
            p1 = float("nan")
        try:
            _, p2 = stats.ttest_rel(primary_per_fold_auprc, sibling_auprc)
        except Exception:
            p2 = float("nan")
        r["p_AUROC_vs_primary"] = float(p1)
        r["p_AUPRC_vs_primary"] = float(p2)
        pvals_auroc[model] = float(p1) if not np.isnan(p1) else float("nan")
        pvals_auprc[model] = float(p2) if not np.isnan(p2) else float("nan")
    holm_a = holm_bonferroni({k: v for k, v in pvals_auroc.items() if not np.isnan(v)})
    holm_b = holm_bonferroni({k: v for k, v in pvals_auprc.items() if not np.isnan(v)})
    for r in rows:
        r["pHolm_AUROC_vs_primary"] = holm_a.get(r["Model"]) if r["Model"] != PRIMARY_MODEL_NAME else None
        r["pHolm_AUPRC_vs_primary"] = holm_b.get(r["Model"]) if r["Model"] != PRIMARY_MODEL_NAME else None

    out_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "structure_ablation.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
