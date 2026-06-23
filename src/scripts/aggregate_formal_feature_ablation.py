"""Aggregate feature-ablation fold pickles into paper-ready CSVs.

For every setting listed in :data:`FORMAL_FEATURE_ABLATIONS`, we look up the
corresponding ``<MODEL>__<setting>_fold<K>.pkl`` files and summarise them.
The primary table reports the **primary model** (``HerbPairIAM``); a
supplementary table reports the ``DoseAwareIAM`` counterpart (kept for
completeness, not used as a claim in the paper); a third supplementary
table reports the ``XGBoost`` tabular variant.

Paired Wilcoxon signed-rank tests (matched across 10 folds) are computed
for every setting vs. the ``full`` reference, followed by
Holm–Bonferroni correction across the number of non-``full`` settings.

Outputs::

    tables/feature_ablation.csv                         # primary: HerbPairIAM
    tables/feature_ablation_doseaware_supplementary.csv # reference: DoseAwareIAM
    tables/feature_ablation_xgboost_supplementary.csv   # reference: XGBoost

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/feature_ablation \\
    python -u src/aggregate_formal_feature_ablation.py
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import pickle

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import FOLD_RESULTS_DIR, TABLES_DIR, holm_bonferroni
from phase4_evaluation import FORMAL_FEATURE_ABLATIONS, PRIMARY_MODEL_NAME
from neural_models import summarize_results as summarize_neural
from tabular_models import summarize_results as summarize_tabular


def _load_tag_results(tag: str) -> list[dict]:
    results = []
    for fold_id in range(10):
        path = FOLD_RESULTS_DIR / f"{tag}_fold{fold_id}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            results.append(pickle.load(f))
    return sorted(results, key=lambda r: int(r["fold"]))


def _paired_stats(
    full_results: list[dict],
    setting_results: list[dict],
    metric_key: str,
) -> tuple[float, float]:
    """Paired Wilcoxon on fold-level metric. Returns (delta, p-value)."""
    if len(full_results) != len(setting_results) or len(full_results) < 2:
        return float("nan"), float("nan")
    a = np.asarray([float(r[metric_key]) for r in setting_results], dtype=float)
    b = np.asarray([float(r[metric_key]) for r in full_results], dtype=float)
    delta = float(a.mean() - b.mean())
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        # wilcoxon raises when the paired differences are all zero (identical
        # pickles, e.g. HerbPairIAM__full hard-linked to HerbPairIAM__without_dose).
        # Return a deterministic p=1.0 since no difference means no evidence.
        p = 1.0
    return delta, float(p)


def _aggregate_model(
    model_name: str,
    settings: list[tuple[str, object]],
    summarise_fn,
) -> pd.DataFrame:
    # First pass: load full's fold-level results for the paired tests.
    full_tag = f"{model_name}__full"
    full_results = _load_tag_results(full_tag)
    rows = []
    raw_p_auroc = {}
    raw_p_auprc = {}
    for setting, _ in settings:
        tag = f"{model_name}__{setting}"
        results = _load_tag_results(tag)
        if not results:
            continue
        summary = summarise_fn(results)
        row = {
            "Model": model_name,
            "Setting": setting,
            "n_folds": int(len(results)),
            "AUROC": summary["auroc_mean"],
            "AUROC_std": summary["auroc_std"],
            "AUPRC": summary["auprc_mean"],
            "AUPRC_std": summary["auprc_std"],
        }
        if setting != "full" and full_results:
            d_au, p_au = _paired_stats(full_results, results, "auroc")
            d_ap, p_ap = _paired_stats(full_results, results, "auprc")
            row["delta_AUROC_vs_full"] = d_au
            row["delta_AUPRC_vs_full"] = d_ap
            row["p_AUROC_vs_full"] = p_au
            row["p_AUPRC_vs_full"] = p_ap
            raw_p_auroc[setting] = p_au
            raw_p_auprc[setting] = p_ap
        else:
            row["delta_AUROC_vs_full"] = 0.0 if setting == "full" else float("nan")
            row["delta_AUPRC_vs_full"] = 0.0 if setting == "full" else float("nan")
            row["p_AUROC_vs_full"] = float("nan")
            row["p_AUPRC_vs_full"] = float("nan")
        rows.append(row)

    # Holm-Bonferroni across non-``full`` settings.
    if raw_p_auroc:
        holm_au = holm_bonferroni(raw_p_auroc)
        holm_ap = holm_bonferroni(raw_p_auprc)
        for row in rows:
            s = row["Setting"]
            row["pHolm_AUROC_vs_full"] = holm_au.get(s, float("nan"))
            row["pHolm_AUPRC_vs_full"] = holm_ap.get(s, float("nan"))
    return pd.DataFrame(rows)


def main():
    # Primary model — this is the table that lands in the paper.
    primary_df = _aggregate_model(PRIMARY_MODEL_NAME, FORMAL_FEATURE_ABLATIONS, summarize_neural)
    if not primary_df.empty:
        out_primary = TABLES_DIR / "feature_ablation.csv"
        primary_df.to_csv(out_primary, index=False)
        print(f"Wrote {out_primary}  (rows={len(primary_df)})")
        print(primary_df.to_string(index=False))

    # Reference with-dose supplementary.
    dose_df = _aggregate_model("DoseAwareIAM", FORMAL_FEATURE_ABLATIONS, summarize_neural)
    if not dose_df.empty:
        out_dose = TABLES_DIR / "feature_ablation_doseaware_supplementary.csv"
        dose_df.to_csv(out_dose, index=False)
        print(f"\nWrote {out_dose}  (rows={len(dose_df)})  [supplementary, not used as primary claim]")

    # Tabular XGBoost supplementary.
    xgb_df = _aggregate_model("XGBoost", FORMAL_FEATURE_ABLATIONS, summarize_tabular)
    if not xgb_df.empty:
        out_xgb = TABLES_DIR / "feature_ablation_xgboost_supplementary.csv"
        xgb_df.to_csv(out_xgb, index=False)
        print(f"\nWrote {out_xgb}  (rows={len(xgb_df)})")


if __name__ == "__main__":
    main()
