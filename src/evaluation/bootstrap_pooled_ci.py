"""Bootstrap 95% confidence intervals on pooled OOF predictions.

Two resampling strategies are supported via ``--bootstrap-mode``:

* ``cluster`` (default, recommended for publication):
  resample the *fold identifiers* with replacement. Each fold's (y_true,
  y_prob) block is kept intact; fold-level dependence between samples is
  therefore respected. This produces a more honest 95% CI that does not
  assume independence across fold-mates. This is the correct estimator for
  our 10-fold CV setting.

* ``iid`` (legacy):
  resample individual (y_true, y_prob) entries with replacement after
  concatenating all folds. Assumes independence across samples, which is
  violated when samples in the same fold share a model and training set.
  Kept only for backward compatibility with earlier outputs.

Both estimators concatenate folds to compute the point estimate (AUROC and
AUPRC on the pooled OOF predictions), so the point estimate is identical
between modes; only the CI differs.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/bootstrap_pooled_ci.py \\
        [--n-boot 2000] [--bootstrap-mode cluster|iid] [--include MODEL,...]

Outputs:
    ``supplementary/pooled_bootstrap_ci.csv``  — one row per model with
        pooled_auroc / pooled_auprc and 95% CI columns.

References:
    * Davison & Hinkley (1997), *Bootstrap Methods and their Application*,
      §3.8 on cluster bootstrap.
    * Ren et al. (2010), *Nonparametric bootstrapping for
      hierarchical data*, Journal of Applied Statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_utils import (
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    compute_pooled_predictions,
    ensure_output_dirs,
    load_pickle,
)


sys.stdout.reconfigure(line_buffering=True)


def _safe_metric(y_t: np.ndarray, y_p: np.ndarray, fn) -> float:
    """AUROC/AUPRC only defined when both classes are present."""
    if np.unique(y_t).size < 2:
        return float("nan")
    try:
        return float(fn(y_t, y_p))
    except Exception:
        return float("nan")


def _bootstrap_ci_iid(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int, seed: int) -> dict:
    """Legacy i.i.d. sample-level bootstrap (kept for backward compat)."""
    rng = np.random.default_rng(seed)
    n = int(len(y_true))
    auroc_pool = np.full(n_boot, np.nan, dtype=float)
    auprc_pool = np.full(n_boot, np.nan, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        auroc_pool[i] = _safe_metric(y_true[idx], y_prob[idx], roc_auc_score)
        auprc_pool[i] = _safe_metric(y_true[idx], y_prob[idx], average_precision_score)
    return _summarise(auroc_pool, auprc_pool)


def _bootstrap_ci_cluster(results: list[dict], n_boot: int, seed: int) -> dict:
    """Cluster-by-fold bootstrap.

    For each resample we draw K fold ids with replacement from {0..K-1}, then
    concatenate the corresponding (y_true, y_prob) blocks. This preserves
    within-fold dependence, producing an honest 95% CI under the standard
    10-fold CV assumption that folds are independent but samples within a
    fold are not.
    """
    rng = np.random.default_rng(seed)
    # Pre-extract each fold's y_true/y_prob once.
    blocks = [
        (np.asarray(r["y_true"], dtype=int), np.asarray(r["y_prob"], dtype=float))
        for r in results
    ]
    k = len(blocks)
    if k < 2:
        # With a single fold, fall back to sample-level bootstrap to still
        # produce a non-degenerate CI, and flag that in the output.
        y_true = np.concatenate([b[0] for b in blocks]) if blocks else np.array([], dtype=int)
        y_prob = np.concatenate([b[1] for b in blocks]) if blocks else np.array([], dtype=float)
        result = _bootstrap_ci_iid(y_true, y_prob, n_boot=n_boot, seed=seed)
        result["bootstrap_mode"] = "cluster->iid_fallback_single_fold"
        return result
    auroc_pool = np.full(n_boot, np.nan, dtype=float)
    auprc_pool = np.full(n_boot, np.nan, dtype=float)
    for i in range(n_boot):
        fold_idx = rng.integers(0, k, size=k)
        y_t = np.concatenate([blocks[j][0] for j in fold_idx])
        y_p = np.concatenate([blocks[j][1] for j in fold_idx])
        auroc_pool[i] = _safe_metric(y_t, y_p, roc_auc_score)
        auprc_pool[i] = _safe_metric(y_t, y_p, average_precision_score)
    result = _summarise(auroc_pool, auprc_pool)
    result["bootstrap_mode"] = "cluster"
    return result


def _summarise(auroc_pool: np.ndarray, auprc_pool: np.ndarray) -> dict:
    auroc_valid = auroc_pool[np.isfinite(auroc_pool)]
    auprc_valid = auprc_pool[np.isfinite(auprc_pool)]
    if auroc_valid.size == 0 or auprc_valid.size == 0:
        return {
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
            "n_boot_used": int(min(auroc_valid.size, auprc_valid.size)),
        }
    auroc_low, auroc_high = np.percentile(auroc_valid, [2.5, 97.5])
    auprc_low, auprc_high = np.percentile(auprc_valid, [2.5, 97.5])
    return {
        "auroc_ci_low": float(auroc_low),
        "auroc_ci_high": float(auroc_high),
        "auprc_ci_low": float(auprc_low),
        "auprc_ci_high": float(auprc_high),
        "n_boot_used": int(min(auroc_valid.size, auprc_valid.size)),
    }


def _bootstrap_ci_pair(
    results: list[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int,
    seed: int,
    mode: str,
) -> dict:
    """Dispatch to the requested bootstrap mode."""
    if mode == "cluster":
        return _bootstrap_ci_cluster(results, n_boot=n_boot, seed=seed)
    if mode == "iid":
        out = _bootstrap_ci_iid(y_true, y_prob, n_boot=n_boot, seed=seed)
        out["bootstrap_mode"] = "iid"
        return out
    raise ValueError(f"Unknown bootstrap mode: {mode!r}. Use 'cluster' or 'iid'.")


def _load_complete_models(min_folds: int = 1) -> dict[str, list[dict]]:
    if not FOLD_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Fold results dir not found: {FOLD_RESULTS_DIR}")
    by_model: dict[str, list[dict]] = {}
    seen_paths = set()
    for pattern in ("*_fold*.pkl", "cold_*.pkl"):
        for path in sorted(FOLD_RESULTS_DIR.glob(pattern)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                obj = load_pickle(path)
            except Exception:
                continue
            if "y_true" not in obj or "y_prob" not in obj:
                continue
            model = str(obj.get("model", path.stem.split("_fold")[0]))
            by_model.setdefault(model, []).append(obj)
    selected = {}
    for model, results in by_model.items():
        if len(results) >= min_folds:
            selected[model] = sorted(results, key=lambda r: int(r.get("fold", 0)))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (default 2000).")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--bootstrap-mode",
        default="cluster",
        choices=["cluster", "iid"],
        help="'cluster' (default): resample fold IDs with replacement; respects "
             "within-fold dependence. 'iid': legacy sample-level bootstrap, "
             "kept for backward compat only.",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Optional comma-separated subset of model names to include.",
    )
    parser.add_argument(
        "--min-folds",
        type=int,
        default=1,
        help="Minimum fold pickles required for a model to be considered.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    by_model = _load_complete_models(min_folds=args.min_folds)
    if args.include:
        keep = {m.strip() for m in args.include.split(",") if m.strip()}
        by_model = {m: r for m, r in by_model.items() if m in keep}
    if not by_model:
        print("No fold pickles available; nothing to bootstrap.")
        return 1

    rows = []
    for model_name, results in sorted(by_model.items()):
        y_true, y_prob = compute_pooled_predictions(results)
        if y_true.size == 0:
            continue
        try:
            auroc_point = float(roc_auc_score(y_true, y_prob))
            auprc_point = float(average_precision_score(y_true, y_prob))
        except Exception:
            auroc_point = float("nan")
            auprc_point = float("nan")
        ci = _bootstrap_ci_pair(
            results=results,
            y_true=y_true,
            y_prob=y_prob,
            n_boot=args.n_boot,
            seed=args.seed + hash(model_name) % 1000,
            mode=args.bootstrap_mode,
        )
        rows.append({
            "Model": model_name,
            "n_pooled_samples": int(y_true.size),
            "n_folds": int(len(results)),
            "pooled_auroc": auroc_point,
            "pooled_auprc": auprc_point,
            **ci,
        })
        print(
            f"  {model_name}: AUROC={auroc_point:.4f} (95% CI [{ci['auroc_ci_low']:.4f},{ci['auroc_ci_high']:.4f}]) "
            f"| AUPRC={auprc_point:.4f} ([{ci['auprc_ci_low']:.4f},{ci['auprc_ci_high']:.4f}]) "
            f"[{ci.get('bootstrap_mode', args.bootstrap_mode)}]"
        )

    boot_df = pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)
    boot_path = SUPP_DIR / "pooled_bootstrap_ci.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"\nWrote {boot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
