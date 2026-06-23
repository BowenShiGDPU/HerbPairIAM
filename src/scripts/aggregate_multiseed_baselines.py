"""Aggregate canonical (seed=42) + multiseed baselines into a 30-fold table.

Reads per-model fold pickles from two subdirectories:

* ``main_benchmark/fold_results/<Model>_fold<K>.pkl``   (seed=42, canonical)
* ``multiseed_baselines/fold_results/<Model>_seed<S>_fold<K>.pkl``
  (additional seeds, typically 13 and 7)

For every model present in both, it concatenates all fold results into a
single 30-fold (or 3-seed) set and reports mean ± seed-std across seeds,
pooled cluster-bootstrap CI, and per-seed summary. Output CSVs live
under ``main_benchmark/tables/`` so the main benchmark table can be
updated in place to report multi-seed CIs.

Also outputs per-seed per-model summary into
``main_benchmark/tables/main_benchmark_multiseed.csv``.

Usage::

    RESULTS_ROOT_DIR=results python -u src/scripts/aggregate_multiseed_baselines.py \\
        --canonical-subdir formal_doseaware_neg10_auroc/main_benchmark \\
        --multiseed-subdir formal_doseaware_neg10_auroc/multiseed_baselines \\
        --canonical-seed 42 \\
        --extra-seeds 13 7
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


sys.stdout.reconfigure(line_buffering=True)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT_DIR = Path(ROOT_DIR / (os_env := __import__("os").environ.get("RESULTS_ROOT_DIR", "results")))
if not RESULTS_ROOT_DIR.is_absolute():
    RESULTS_ROOT_DIR = ROOT_DIR / RESULTS_ROOT_DIR

_FOLD_RE_CANON = re.compile(r"^(?P<model>.+)_fold(?P<fold>\d+)$")
_FOLD_RE_MULTI = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)_fold(?P<fold>\d+)$")


def _load_canonical(canon_dir: Path, canonical_seed: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not canon_dir.exists():
        return out
    for p in sorted(canon_dir.glob("*_fold*.pkl")):
        m = _FOLD_RE_CANON.match(p.stem)
        if not m:
            continue
        try:
            obj = pickle.load(open(p, "rb"))
        except Exception:
            continue
        if "y_true" not in obj or "y_prob" not in obj:
            continue
        obj["outer_seed"] = canonical_seed
        out.setdefault(m.group("model"), []).append(obj)
    return out


def _load_multiseed(multi_dir: Path, extra_seeds: list[int]) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    if not multi_dir.exists():
        return out
    for p in sorted(multi_dir.glob("*_seed*_fold*.pkl")):
        m = _FOLD_RE_MULTI.match(p.stem)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed not in extra_seeds:
            continue
        try:
            obj = pickle.load(open(p, "rb"))
        except Exception:
            continue
        if "y_true" not in obj or "y_prob" not in obj:
            continue
        obj["outer_seed"] = seed
        model = m.group("model")
        out.setdefault(model, {}).setdefault(seed, []).append(obj)
    return out


def _mean_std(vals: np.ndarray) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def _cluster_bootstrap_ci(results_list: list[list[dict]], n_boot: int = 2000, seed: int = 42) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap (y_true, y_prob) at the fold level.

    ``results_list`` is a list whose elements are lists of fold dicts (one
    per seed). We treat every fold (regardless of seed) as an independent
    cluster for resampling purposes — this is the standard pooled
    cluster-bootstrap CI reported in Nature Commun ML papers.
    """
    all_folds = [fold for per_seed in results_list for fold in per_seed]
    if not all_folds:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    k = len(all_folds)
    au_pool = np.full(n_boot, np.nan)
    ap_pool = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        y_t = np.concatenate([np.asarray(all_folds[j]["y_true"], dtype=int) for j in idx])
        y_p = np.concatenate([np.asarray(all_folds[j]["y_prob"], dtype=float) for j in idx])
        if np.unique(y_t).size >= 2:
            au_pool[i] = roc_auc_score(y_t, y_p)
            ap_pool[i] = average_precision_score(y_t, y_p)
    au_ci = tuple(np.percentile(au_pool[np.isfinite(au_pool)], [2.5, 97.5])) if np.any(np.isfinite(au_pool)) else (float("nan"), float("nan"))
    ap_ci = tuple(np.percentile(ap_pool[np.isfinite(ap_pool)], [2.5, 97.5])) if np.any(np.isfinite(ap_pool)) else (float("nan"), float("nan"))
    return au_ci, ap_ci


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-subdir", required=True,
                        help="e.g. formal_doseaware_neg10_auroc/main_benchmark")
    parser.add_argument("--multiseed-subdir", required=True,
                        help="e.g. formal_doseaware_neg10_auroc/multiseed_baselines")
    parser.add_argument("--canonical-seed", type=int, default=42)
    parser.add_argument("--extra-seeds", type=int, nargs="+", default=[13, 7])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    canon_dir = RESULTS_ROOT_DIR / args.canonical_subdir / "fold_results"
    multi_dir = RESULTS_ROOT_DIR / args.multiseed_subdir / "fold_results"
    out_tables_dir = RESULTS_ROOT_DIR / args.canonical_subdir / "tables"
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    canon = _load_canonical(canon_dir, args.canonical_seed)
    multi = _load_multiseed(multi_dir, args.extra_seeds)

    all_models = sorted(set(canon) | set(multi))

    # Per-seed rows.
    per_seed_rows = []
    # 30-fold summary rows.
    summary_rows = []
    for model in all_models:
        per_seed_results: dict[int, list[dict]] = {}
        # Canonical seed.
        if model in canon:
            per_seed_results[args.canonical_seed] = sorted(canon[model], key=lambda r: int(r.get("fold", 0)))
        # Extra seeds.
        if model in multi:
            for s, rs in multi[model].items():
                per_seed_results[s] = sorted(rs, key=lambda r: int(r.get("fold", 0)))

        seed_metrics = {}
        for s, rs in per_seed_results.items():
            if len(rs) < 10:
                continue
            auroc = np.array([r["auroc"] for r in rs], dtype=float)
            auprc = np.array([r["auprc"] for r in rs], dtype=float)
            seed_metrics[s] = (auroc.mean(), auprc.mean())
            per_seed_rows.append({
                "model": model, "seed": int(s), "n_folds": len(rs),
                "auroc_mean": float(auroc.mean()), "auroc_std": float(auroc.std(ddof=0)),
                "auprc_mean": float(auprc.mean()), "auprc_std": float(auprc.std(ddof=0)),
            })

        if not seed_metrics:
            continue
        au_means = np.array([v[0] for v in seed_metrics.values()])
        ap_means = np.array([v[1] for v in seed_metrics.values()])
        # Pooled CI across all folds from all seeds.
        results_list = [per_seed_results.get(s, []) for s in per_seed_results]
        au_ci, ap_ci = _cluster_bootstrap_ci(results_list, n_boot=args.n_boot, seed=42 + hash(model) % 1000)
        summary_rows.append({
            "model": model,
            "n_seeds": len(seed_metrics),
            "seeds": ",".join(str(s) for s in sorted(seed_metrics.keys())),
            "n_folds_total": sum(len(rs) for rs in per_seed_results.values()),
            "auroc_seed_mean": float(au_means.mean()),
            "auroc_seed_std": float(au_means.std(ddof=0)) if len(au_means) > 1 else 0.0,
            "auprc_seed_mean": float(ap_means.mean()),
            "auprc_seed_std": float(ap_means.std(ddof=0)) if len(ap_means) > 1 else 0.0,
            "auroc_ci_low": float(au_ci[0]), "auroc_ci_high": float(au_ci[1]),
            "auprc_ci_low": float(ap_ci[0]), "auprc_ci_high": float(ap_ci[1]),
        })

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["model", "seed"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("auprc_seed_mean", ascending=False).reset_index(drop=True)

    per_seed_out = out_tables_dir / "main_benchmark_per_seed.csv"
    summary_out = out_tables_dir / "main_benchmark_multiseed.csv"
    per_seed_df.to_csv(per_seed_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote {per_seed_out}  rows={len(per_seed_df)}")
    print(f"Wrote {summary_out}  rows={len(summary_df)}")
    print()
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
