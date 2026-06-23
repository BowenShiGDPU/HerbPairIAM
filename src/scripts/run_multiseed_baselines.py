"""Multi-seed baseline runs for Nature-Communications-grade CI.

For each requested ``outer_seed`` we construct a fresh 10-fold
pair-stratified split (same mechanics as ``phase2_dataset.build_fold_splits``
but seeded from the CLI), then train each requested baseline on every
fold. Fold pickles are written under
``EXPERIMENT_SUBDIR/fold_results/<Model>_seed<S>_fold<K>.pkl``.

This adds extra seeds without touching the canonical seed=42 fold pickles
that already live under ``main_benchmark/fold_results/``. Aggregation is a
separate step (see ``aggregate_multiseed_baselines.py``) that pools
seed=42 from the canonical run with the new seeds here.

Tabular baselines (LogisticRegression, RandomForest, GradientBoosting,
MLP, XGBoost) use the same grid-search protocol as the canonical run:
hyperparameters are selected once on the first fold of each seed via
``search_params`` and reused across the remaining folds. We keep the same
protocol as seed=42 to maximise comparability; a nested-CV variant is
explored separately in :mod:`run_nested_cv_baselines.py` (Phase 5.1).

Graph baselines (R-GCN, HGT) use the frozen ``GraphConfig`` defaults.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/multiseed_baselines \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/scripts/run_multiseed_baselines.py \\
        --seeds 13 7 \\
        --models XGBoost RandomForest GradientBoosting LogisticRegression MLP R-GCN HGT \\
        --neg-ratio 10
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from data.phase2_dataset import INNER_VAL_FRAC, OUTER_FOLDS
from experiment_utils import FOLD_RESULTS_DIR, load_pickle, sanitize_name, save_pickle
from models.graph_baselines import GraphConfig
from models.graph_baselines import train_one_split as graph_train_split
from models.tabular_models import MODEL_GRIDS, fit_predict_split, search_params
from phase4_evaluation import prepare_common_inputs


sys.stdout.reconfigure(line_buffering=True)


TABULAR_MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP", "XGBoost"]
GRAPH_MODELS = ["R-GCN", "HGT"]
SUPPORTED_MODELS = TABULAR_MODELS + GRAPH_MODELS


def build_fold_splits_with_seed(df: pd.DataFrame, outer_seed: int) -> list[dict]:
    """Reproduce ``phase2_dataset.build_fold_splits`` with a controllable seed."""
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    splits = []
    for fold_id, (train_val_idx, test_idx) in enumerate(
        outer.split(df.index.values, df["label"].values)
    ):
        train_val_idx = np.asarray(train_val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_FRAC,
            random_state=outer_seed + fold_id,
        )
        inner_train, inner_val = next(
            inner.split(train_val_idx, df.iloc[train_val_idx]["label"].values)
        )
        train_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]
        splits.append(
            {
                "fold": fold_id,
                "outer_seed": outer_seed,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    return splits


def _fold_path(tag: str, fold_id: int):
    return FOLD_RESULTS_DIR / f"{sanitize_name(tag)}_fold{fold_id}.pkl"


def _run_tabular(model_name: str, X, labels, splits: list[dict], neg_ratio: int, seed: int):
    tag = f"{model_name}_seed{seed}"
    # Hyperparameter search on the first fold only, matching the canonical
    # seed=42 protocol. Bit-exact reproducibility requires the same fold-0
    # hyperparameters to be reused for all outer folds of this seed.
    params = search_params(model_name, X, labels, splits[0], neg_ratio=neg_ratio)
    print(f"  [{tag}] best params on fold 0: {params}", flush=True)
    for split in splits:
        fold_id = int(split["fold"])
        path = _fold_path(tag, fold_id)
        if path.exists():
            print(f"  [{tag}] fold {fold_id}: skip (pkl exists)", flush=True)
            continue
        result = fit_predict_split(
            model_name, X, labels, split, params,
            neg_ratio=neg_ratio, seed=42, feature_idx=None,
        )
        result["outer_seed"] = seed
        result["hyperparams"] = params
        save_pickle(result, path)
        print(f"  [{tag}] fold {fold_id}: AUROC={result['auroc']:.4f} AUPRC={result['auprc']:.4f}", flush=True)


def _run_graph(model_name: str, ds: dict, splits: list[dict], neg_ratio: int, seed: int):
    tag = f"{model_name}_seed{seed}"
    cfg = GraphConfig(neg_ratio=neg_ratio)
    for split in splits:
        fold_id = int(split["fold"])
        path = _fold_path(tag, fold_id)
        if path.exists():
            print(f"  [{tag}] fold {fold_id}: skip (pkl exists)", flush=True)
            continue
        # Inject the specific split into ds by swapping ``fold_splits``.
        # graph_train_split reads the split dict directly, not the full ds,
        # so we pass it through.
        result = graph_train_split(model_name, ds, split, cfg, save_result=False)
        result["outer_seed"] = seed
        save_pickle(result, path)
        print(f"  [{tag}] fold {fold_id}: AUROC={result['auroc']:.4f} AUPRC={result['auprc']:.4f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Outer CV seeds to add.")
    parser.add_argument("--models", nargs="+", default=SUPPORTED_MODELS,
                        help=f"Baselines to train. Defaults to all of {SUPPORTED_MODELS}.")
    parser.add_argument("--neg-ratio", type=int, default=10)
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in SUPPORTED_MODELS]
    if unknown:
        parser.error(f"Unsupported model(s): {unknown}. Supported: {SUPPORTED_MODELS}")

    ds, df, _, X, labels, hp, ap, pf, lookups = prepare_common_inputs()

    for seed in args.seeds:
        splits = build_fold_splits_with_seed(df, seed)
        print(f"\n===== outer_seed={seed}  folds={[int(s['fold']) for s in splits]} =====", flush=True)
        for model_name in args.models:
            if model_name in TABULAR_MODELS:
                _run_tabular(model_name, X, labels, splits, args.neg_ratio, seed)
            elif model_name in GRAPH_MODELS:
                # graph baselines need ``fold_splits`` reference for some
                # edge-type construction; pass a shallow copy of ds with the
                # current seed's splits inserted.
                ds_with_splits = dict(ds); ds_with_splits["fold_splits"] = splits
                _run_graph(model_name, ds_with_splits, splits, args.neg_ratio, seed)
            else:
                parser.error(f"Unsupported model: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
