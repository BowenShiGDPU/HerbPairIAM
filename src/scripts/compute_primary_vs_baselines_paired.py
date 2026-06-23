"""Paired Wilcoxon: HerbPairIAM vs each baseline on the full 30-fold set.

For every (outer_seed, fold) pair we have:
  * HerbPairIAM fold pickle    — from head2head (seed 42/13/7)
  * baseline fold pickle        — from main_benchmark (seed=42, canonical)
                                  and multiseed_baselines (seed=13, 7)

We align the (seed, fold) pairs and run a paired Wilcoxon signed-rank
test on fold-level AUROC and AUPRC. Holm-Bonferroni correction is
applied across the K baseline comparisons.

This is the fold-level paired counterpart of the DeLong pair-wise test
that is computed on pooled OOF predictions. Both are reported in the
paper: DeLong for the pooled AUROC claim, paired Wilcoxon for the
fold-level "consistent improvement" claim.

Outputs::

    supplementary/primary_vs_baselines_paired_30fold.csv

Usage::

    python -u src/scripts/compute_primary_vs_baselines_paired.py
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "formal_doseaware_neg10_auroc"


# --- Fold-pkl source map ----------------------------------------------------
# HerbPairIAM lives in head2head under ``H2H_seed<S>_HerbPairIAM_fold<F>.pkl``.
# Canonical HerbPairIAM (main_benchmark seed=42) is separately present but
# duplicates the seed=42 head-to-head pickles; we use head2head for symmetry.
def _load_primary_30fold() -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for p in sorted((RESULTS_DIR / "dose_head2head" / "fold_results").glob("H2H_seed*_HerbPairIAM_fold*.pkl")):
        m = re.match(r"H2H_seed(\d+)_HerbPairIAM_fold(\d+)\.pkl", p.name)
        if not m:
            continue
        seed, fold = int(m.group(1)), int(m.group(2))
        with open(p, "rb") as f:
            out[(seed, fold)] = pickle.load(f)
    return out


def _load_baseline_30fold(model_name: str) -> dict[tuple[int, int], dict]:
    """Resolve 30 baseline fold pickles from (main_benchmark seed=42) +
    (multiseed_baselines seed=13, seed=7)."""
    out: dict[tuple[int, int], dict] = {}

    # seed=42 — canonical main_benchmark. Naming: ``<Model>_fold<F>.pkl``.
    for p in sorted((RESULTS_DIR / "main_benchmark" / "fold_results").glob(f"{model_name}_fold*.pkl")):
        m = re.match(rf"{re.escape(model_name)}_fold(\d+)\.pkl", p.name)
        if not m:
            continue
        fold = int(m.group(1))
        with open(p, "rb") as f:
            out[(42, fold)] = pickle.load(f)

    # seed=13 and seed=7 — multiseed_baselines.
    # Naming: ``<Model>_seed<S>_fold<F>.pkl``.
    for p in sorted((RESULTS_DIR / "multiseed_baselines" / "fold_results").glob(f"{model_name}_seed*_fold*.pkl")):
        m = re.match(rf"{re.escape(model_name)}_seed(\d+)_fold(\d+)\.pkl", p.name)
        if not m:
            continue
        seed, fold = int(m.group(1)), int(m.group(2))
        with open(p, "rb") as f:
            out[(seed, fold)] = pickle.load(f)
    return out


def _holm_bonferroni(p_dict: dict[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values keyed the same as the input."""
    if not p_dict:
        return {}
    items = sorted(p_dict.items(), key=lambda kv: kv[1])
    k = len(items)
    adj = {}
    running_max = 0.0
    for rank, (name, raw_p) in enumerate(items):
        mult = k - rank
        val = min(1.0, raw_p * mult)
        # Enforce monotonicity of Holm-adjusted p-values.
        running_max = max(running_max, val)
        adj[name] = running_max
    return adj


def main() -> int:
    primary = _load_primary_30fold()
    if len(primary) < 30:
        print(f"[warn] expected 30 HerbPairIAM folds; got {len(primary)}")

    BASELINES = [
        "InteractionAwareSetModel",
        "IAM_Wide",
        "DoseAwareIAM",
        "XGBoost",
        "GradientBoosting",
        "RandomForest",
        "LogisticRegression",
        "MLP",
        "R-GCN",
        "HGT",
    ]

    raw_p_auroc: dict[str, float] = {}
    raw_p_auprc: dict[str, float] = {}
    rows = []
    for model in BASELINES:
        baseline = _load_baseline_30fold(model)
        keys = sorted(set(primary) & set(baseline))
        if not keys:
            print(f"[skip] no paired folds for {model}")
            continue
        p_au = np.array([primary[k]["auroc"] for k in keys], dtype=float)
        b_au = np.array([baseline[k]["auroc"] for k in keys], dtype=float)
        p_ap = np.array([primary[k]["auprc"] for k in keys], dtype=float)
        b_ap = np.array([baseline[k]["auprc"] for k in keys], dtype=float)
        try:
            _, pW_au = stats.wilcoxon(p_au, b_au)
        except ValueError:
            pW_au = float("nan")
        try:
            _, pW_ap = stats.wilcoxon(p_ap, b_ap)
        except ValueError:
            pW_ap = float("nan")
        try:
            _, pT_au = stats.ttest_rel(p_au, b_au)
        except Exception:
            pT_au = float("nan")
        try:
            _, pT_ap = stats.ttest_rel(p_ap, b_ap)
        except Exception:
            pT_ap = float("nan")
        win_au = int((p_au > b_au).sum())
        win_ap = int((p_ap > b_ap).sum())
        rows.append({
            "baseline": model,
            "n_paired": len(keys),
            "HerbPairIAM_auroc_mean": float(p_au.mean()),
            "baseline_auroc_mean": float(b_au.mean()),
            "delta_auroc_mean": float(p_au.mean() - b_au.mean()),
            "delta_auroc_se": float(np.std(p_au - b_au, ddof=1) / np.sqrt(len(keys))),
            "auroc_wins": win_au,
            "wilcoxon_p_auroc": float(pW_au),
            "ttest_p_auroc": float(pT_au),
            "HerbPairIAM_auprc_mean": float(p_ap.mean()),
            "baseline_auprc_mean": float(b_ap.mean()),
            "delta_auprc_mean": float(p_ap.mean() - b_ap.mean()),
            "delta_auprc_se": float(np.std(p_ap - b_ap, ddof=1) / np.sqrt(len(keys))),
            "auprc_wins": win_ap,
            "wilcoxon_p_auprc": float(pW_ap),
            "ttest_p_auprc": float(pT_ap),
        })
        raw_p_auroc[model] = float(pW_au)
        raw_p_auprc[model] = float(pW_ap)

    holm_au = _holm_bonferroni(raw_p_auroc)
    holm_ap = _holm_bonferroni(raw_p_auprc)
    for row in rows:
        row["wilcoxon_holm_p_auroc"] = holm_au.get(row["baseline"], float("nan"))
        row["wilcoxon_holm_p_auprc"] = holm_ap.get(row["baseline"], float("nan"))

    df = pd.DataFrame(rows).sort_values("delta_auprc_mean", ascending=False).reset_index(drop=True)
    out_dir = RESULTS_DIR / "main_benchmark" / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "primary_vs_baselines_paired_30fold.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(df)}")
    print()
    preview = df[[
        "baseline", "n_paired", "delta_auroc_mean", "auroc_wins",
        "wilcoxon_p_auroc", "wilcoxon_holm_p_auroc",
        "delta_auprc_mean", "auprc_wins",
        "wilcoxon_p_auprc", "wilcoxon_holm_p_auprc",
    ]]
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
