"""Promote HerbEmbIAM to Main Table 2 (structure ablation) as a representation row.

HerbEmbIAM replaces the 12-channel KG-derived pair/herb feature tensor with a
learnable herb-embedding lookup. We already have seed=42 × 10-fold predictions
from the head-to-head run; this script converts them into a table row aligned
with the existing structure-ablation protocol (mean $\\pm$ std, $\\Delta$
vs.\\ HerbPairIAM, paired two-sided Wilcoxon with Holm correction).

Because the table now has 9 ablation settings (instead of 8), all existing
Holm-corrected p-values are recomputed over the expanded family. Raw
Wilcoxon p-values are unchanged.

Inputs
------
- ``results/formal_doseaware_neg10_auroc/main_benchmark/fold_results/HerbPairIAM_fold{0-9}.pkl``
  (canonical seed=42 primary)
- ``results/formal_doseaware_neg10_auroc/dose_head2head/fold_results/H2H_seed42_HerbEmbIAM_fold{0-9}.pkl``
  (HerbEmbIAM seed=42)
- ``paper_package/main/tables/table1_structure_ablation.csv``

Outputs
-------
- Overwrites ``paper_package/main/tables/table1_structure_ablation.csv`` with
  a new ``HerbEmbIAM`` row appended and all Holm p-values recomputed.
"""

from __future__ import annotations

import pathlib as _pathlib
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = _pathlib.Path(__file__).resolve().parent.parent.parent
CANON = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "fold_results"
H2H = ROOT / "results" / "formal_doseaware_neg10_auroc" / "dose_head2head" / "fold_results"
TABLE_PATH = ROOT / "paper_package" / "main" / "tables" / "table1_structure_ablation.csv"


def _load_primary_per_fold():
    rows = []
    for k in range(10):
        with open(CANON / f"HerbPairIAM_fold{k}.pkl", "rb") as fh:
            r = pickle.load(fh)
        rows.append({
            "fold": k,
            "auroc": roc_auc_score(r["y_true"], r["y_prob"]),
            "auprc": average_precision_score(r["y_true"], r["y_prob"]),
        })
    return pd.DataFrame(rows)


def _load_herbemb_per_fold():
    rows = []
    for k in range(10):
        with open(H2H / f"H2H_seed42_HerbEmbIAM_fold{k}.pkl", "rb") as fh:
            r = pickle.load(fh)
        rows.append({
            "fold": k,
            "auroc": roc_auc_score(r["y_true"], r["y_prob"]),
            "auprc": average_precision_score(r["y_true"], r["y_prob"]),
        })
    return pd.DataFrame(rows)


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, NaN-safe and monotone."""
    arr = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(arr)
    m = int(mask.sum())
    out = np.full_like(arr, np.nan, dtype=float)
    if m == 0:
        return out.tolist()
    idxs = np.where(mask)[0]
    order = idxs[np.argsort(arr[mask])]
    running_max = 0.0
    for step, j in enumerate(order):
        adj = min(1.0, (m - step) * arr[j])
        running_max = max(running_max, adj)
        out[j] = running_max
    return out.tolist()


def main() -> int:
    primary = _load_primary_per_fold().sort_values("fold").reset_index(drop=True)
    herbemb = _load_herbemb_per_fold().sort_values("fold").reset_index(drop=True)
    assert (primary["fold"].values == herbemb["fold"].values).all()

    dfT = pd.read_csv(TABLE_PATH)
    if "HerbEmbIAM" in dfT["model"].values:
        dfT = dfT[dfT["model"] != "HerbEmbIAM"].reset_index(drop=True)

    delta_auroc = herbemb["auroc"].values - primary["auroc"].values
    delta_auprc = herbemb["auprc"].values - primary["auprc"].values
    p_auroc = float(stats.wilcoxon(delta_auroc, alternative="two-sided", zero_method="wilcox").pvalue)
    p_auprc = float(stats.wilcoxon(delta_auprc, alternative="two-sided", zero_method="wilcox").pvalue)

    new_row = {
        "model": "HerbEmbIAM",
        "n_folds": 10,
        "auroc": round(float(herbemb["auroc"].mean()), 4),
        "auprc": round(float(herbemb["auprc"].mean()), 4),
        "auroc_std": round(float(herbemb["auroc"].std(ddof=1)), 4),
        "auprc_std": round(float(herbemb["auprc"].std(ddof=1)), 4),
        "delta_auroc_vs_primary": round(float(delta_auroc.mean()), 4),
        "delta_auprc_vs_primary": round(float(delta_auprc.mean()), 4),
        "p_auroc_vs_primary": p_auroc,
        "p_auprc_vs_primary": p_auprc,
        "p_holm_auroc_vs_primary": np.nan,
        "p_holm_auprc_vs_primary": np.nan,
    }
    dfT = pd.concat([dfT, pd.DataFrame([new_row])], ignore_index=True)

    variant_mask = dfT["model"].str.lower() != "herbpairiam"
    for raw_col, holm_col in [
        ("p_auroc_vs_primary", "p_holm_auroc_vs_primary"),
        ("p_auprc_vs_primary", "p_holm_auprc_vs_primary"),
    ]:
        sub = dfT.loc[variant_mask, raw_col].tolist()
        adj = _holm(sub)
        dfT.loc[variant_mask, holm_col] = adj

    metric_cols = {
        "auroc", "auprc", "auroc_std", "auprc_std",
        "delta_auroc_vs_primary", "delta_auprc_vs_primary",
    }
    for c in metric_cols:
        if c in dfT.columns:
            dfT[c] = dfT[c].astype(float).round(4)
    for c in [
        "p_auroc_vs_primary", "p_auprc_vs_primary",
        "p_holm_auroc_vs_primary", "p_holm_auprc_vs_primary",
    ]:
        if c in dfT.columns:
            def _fmt(v):
                if pd.isna(v):
                    return ""
                v = float(v)
                if v == 0:
                    return "0"
                return f"{v:.3g}"
            dfT[c] = dfT[c].map(_fmt)

    dfT.to_csv(TABLE_PATH, index=False)
    print(f"[add-herbemb-row] wrote {TABLE_PATH}  ({len(dfT)} rows)", flush=True)
    print(dfT[["model", "auroc", "auprc",
               "delta_auroc_vs_primary", "delta_auprc_vs_primary",
               "p_holm_auroc_vs_primary", "p_holm_auprc_vs_primary"]].to_string(index=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
