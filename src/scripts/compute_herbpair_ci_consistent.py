"""Compute a CI that is CONSISTENT with the reported seed-level-mean point estimate.

The existing ``aggregate_multiseed_baselines.py`` reports

  - point estimate = mean of three seed-level means of 10 per-fold AUROC/AUPRC
  - CI            = cluster-bootstrap on *pooled-OOF* AUROC/AUPRC

These are different estimators; the pooled AUPRC can sit below the
fold-averaged AUPRC whenever per-fold prevalence varies, which is why
the previously reported AUPRC CI ``[0.475, 0.512]`` did not bracket
the reported mean ``0.517``. This script recomputes the CI as a
percentile bootstrap on the **fold-averaged** AUROC/AUPRC estimator so
that CI and point estimate are consistent.

Usage::

    python src/scripts/compute_herbpair_ci_consistent.py
"""

from __future__ import annotations

import pathlib as _pathlib
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = _pathlib.Path(__file__).resolve().parent.parent.parent
CANON = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "fold_results"
MULTI = ROOT / "results" / "formal_doseaware_neg10_auroc" / "multiseed_baselines" / "fold_results"


def _load_per_fold_auroc_auprc(model: str) -> list[tuple[int, int, float, float]]:
    rows: list[tuple[int, int, float, float]] = []
    for k in range(10):
        with open(CANON / f"{model}_fold{k}.pkl", "rb") as fh:
            r = pickle.load(fh)
        rows.append((42, k,
                     roc_auc_score(r["y_true"], r["y_prob"]),
                     average_precision_score(r["y_true"], r["y_prob"])))
    for seed in (13, 7):
        for k in range(10):
            p = MULTI / f"{model}_seed{seed}_fold{k}.pkl"
            with open(p, "rb") as fh:
                r = pickle.load(fh)
            rows.append((seed, k,
                         roc_auc_score(r["y_true"], r["y_prob"]),
                         average_precision_score(r["y_true"], r["y_prob"])))
    return rows


def _fold_mean_ci(values: np.ndarray, n_boot: int = 10000, rng_seed: int = 42) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean of a 1-D array.

    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(values[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def main() -> int:
    model = "HerbPairIAM"
    rows = _load_per_fold_auroc_auprc(model)
    arr_auroc = np.array([r[2] for r in rows])
    arr_auprc = np.array([r[3] for r in rows])

    auroc_pt, auroc_lo, auroc_hi = _fold_mean_ci(arr_auroc, rng_seed=42)
    auprc_pt, auprc_lo, auprc_hi = _fold_mean_ci(arr_auprc, rng_seed=43)

    print(f"{model}  n_folds={len(arr_auroc)}  (3 seeds x 10 folds)", flush=True)
    print(f"  AUROC  mean={auroc_pt:.4f}  "
          f"95% bootstrap CI [{auroc_lo:.4f}, {auroc_hi:.4f}]  "
          f"fold_std={arr_auroc.std(ddof=1):.4f}")
    print(f"  AUPRC  mean={auprc_pt:.4f}  "
          f"95% bootstrap CI [{auprc_lo:.4f}, {auprc_hi:.4f}]  "
          f"fold_std={arr_auprc.std(ddof=1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
