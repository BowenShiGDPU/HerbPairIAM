"""Phase 5.1 — Nested CV hyperparameter search for the strongest tabular baseline.

Reviewer-proofing experiment for Nature Communications. The main
``main_benchmark`` run uses a single hyperparameter grid search on the
first outer fold and reuses those hyperparameters across all 10 folds
— a common but optimistic shortcut. Here we compute the *nested-CV*
equivalent for XGBoost: on every outer fold (i.e. every ``train_idx``
partition), we perform an inner stratified 5-fold CV over
``MODEL_GRIDS['XGBoost']`` and pick the hyperparameters that maximise the
inner-average AUROC. The chosen hyperparameters are then used to fit on
the full outer ``train_idx`` and evaluate on the outer ``test_idx``.

This guards against the "fold-0 leakage" concern: no test fold ever
contributes to its own hyperparameter selection. Paper narrative:

  > "For XGBoost — our strongest tabular baseline — we additionally
  > report nested-CV results where hyperparameters are selected on an
  > inner 5-fold split of each outer training fold. The difference to
  > the fold-0-grid protocol used throughout the main benchmark was
  > <= X·X AUROC across all 10 folds, confirming that single-fold
  > selection does not materially inflate our comparison."

Only XGBoost is chosen because
(a) it is our strongest tabular baseline,
(b) its training cost is negligible (seconds per fit), making nested CV
    tractable,
(c) the other tabular baselines cannot close the gap to HerbPairIAM
    anyway, so a nested-CV difference there would not change any claim.

Outputs::

    supplementary/nested_cv_xgboost.csv

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/scripts/run_nested_cv_baseline.py --seeds 42 13 7
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data.phase2_dataset import INNER_VAL_FRAC, OUTER_FOLDS
from experiment_utils import SUPP_DIR, ensure_output_dirs
from models.tabular_models import MODEL_GRIDS, fit_predict_split
from phase4_evaluation import prepare_common_inputs


sys.stdout.reconfigure(line_buffering=True)


def _build_fold_splits(df, outer_seed: int) -> list[dict]:
    from sklearn.model_selection import StratifiedShuffleSplit

    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    splits = []
    for fold_id, (train_val_idx, test_idx) in enumerate(
        outer.split(df.index.values, df["label"].values)
    ):
        train_val_idx = np.asarray(train_val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        inner = StratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                       random_state=outer_seed + fold_id)
        inner_train, inner_val = next(
            inner.split(train_val_idx, df.iloc[train_val_idx]["label"].values)
        )
        train_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]
        splits.append({"fold": fold_id, "outer_seed": outer_seed,
                       "train_idx": train_idx.tolist(),
                       "val_idx": val_idx.tolist(),
                       "test_idx": test_idx.tolist()})
    return splits


def _inner_cv_score(model_name: str, X, y, train_idx: np.ndarray, params: dict,
                    neg_ratio: int, n_inner: int = 5, seed: int = 42) -> float:
    """Average inner-AUROC across ``n_inner`` stratified folds of ``train_idx``.

    We keep a validation holdout inside each inner fold (using the same
    INNER_VAL_FRAC as the outer split) so the tabular early-stopping /
    calibration logic in ``fit_predict_split`` has something to consume.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    scores = []
    for inner_train, inner_test in kf.split(train_idx, y[train_idx]):
        tr_tv = train_idx[inner_train]
        te = train_idx[inner_test]
        # carve out a small inner-val from tr_tv for threshold selection
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        (t2, v2), = sss.split(tr_tv, y[tr_tv])
        split = {
            "fold": -1,
            "train_idx": tr_tv[t2].tolist(),
            "val_idx":   tr_tv[v2].tolist(),
            "test_idx":  te.tolist(),
        }
        res = fit_predict_split(model_name, X, y, split, params,
                                neg_ratio=neg_ratio, seed=seed, feature_idx=None)
        scores.append(float(res["auroc"]))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="XGBoost",
                        help="Tabular model to evaluate with nested CV (default XGBoost).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7])
    parser.add_argument("--neg-ratio", type=int, default=10)
    parser.add_argument("--n-inner", type=int, default=5,
                        help="Inner CV folds per outer fold (default 5).")
    parser.add_argument("--output-name", default="nested_cv_xgboost.csv")
    args = parser.parse_args()
    ensure_output_dirs()

    if args.model not in MODEL_GRIDS:
        parser.error(f"Unknown tabular model: {args.model}")

    ds, df, _, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    grid = MODEL_GRIDS[args.model]
    print(f"[nested_cv] model={args.model}  grid_size={len(grid)}  seeds={args.seeds}")

    rows = []
    for seed in args.seeds:
        splits = _build_fold_splits(df, seed)
        print(f"\n===== outer_seed={seed} =====")
        for split in splits:
            fold_id = int(split["fold"])
            train_idx = np.asarray(split["train_idx"], dtype=int)
            # Inner CV scan the grid.
            best_params, best_score = None, -np.inf
            t0 = time.time()
            for params in grid:
                score = _inner_cv_score(args.model, X, labels, train_idx, params,
                                        neg_ratio=args.neg_ratio,
                                        n_inner=args.n_inner, seed=42)
                if score > best_score:
                    best_score, best_params = score, params
            inner_sec = time.time() - t0
            # Outer fit with best params.
            final = fit_predict_split(args.model, X, labels, split, best_params,
                                      neg_ratio=args.neg_ratio, seed=42,
                                      feature_idx=None)
            rows.append({
                "model": args.model, "seed": seed, "fold": fold_id,
                "best_params": best_params,
                "inner_cv_auroc": best_score,
                "outer_auroc": final["auroc"],
                "outer_auprc": final["auprc"],
                "inner_search_sec": inner_sec,
            })
            print(f"  [{seed}/{fold_id}] inner_cv_auroc={best_score:.4f} "
                  f"outer_auroc={final['auroc']:.4f} outer_auprc={final['auprc']:.4f} "
                  f"(inner search {inner_sec:.1f}s) best={best_params}")

    out_df = pd.DataFrame(rows)
    out_path = SUPP_DIR / args.output_name
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  rows={len(out_df)}")

    # Quick top-level summary.
    summary = out_df.groupby(["model", "seed"]).agg(
        n=("fold", "count"),
        outer_auroc_mean=("outer_auroc", "mean"),
        outer_auroc_std=("outer_auroc", "std"),
        outer_auprc_mean=("outer_auprc", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
