"""Re-train HGT with multiple random restarts per fold, select by validation AUROC.

Rationale
---------
Heterogeneous Graph Transformers are known to be initialisation-sensitive
at modest sample sizes. In the three outer-CV seeds we had, HGT's
ten-fold AUROC was 0.714 / 0.644 / 0.637 — a seed-level std an order of
magnitude larger than every other baseline. Two of the three seeds
clearly landed in poor local optima rather than reflecting the model's
true capacity.

The standard GNN-benchmarking remedy is multiple random restarts with
validation-AUROC selection, not held-out-test tuning. We therefore
keep the original ``GraphConfig`` (same capacity, optimiser, schedule
as reported in the methods) and run ``N_RESTARTS=3`` restarts per fold,
each differing only in the initialisation seed. The restart with the
highest *validation* AUROC is written to the canonical fold pickle.
Test metrics are therefore unbiased: the test split is never looked at
during selection.

Existing HGT fold pickles in main_benchmark (seed=42) and
multiseed_baselines (seeds=13,7) are overwritten in place. Running the
script after completion re-runs from scratch (it always writes).

Usage::

    python src/scripts/rerun_hgt_stable.py --seeds 42 13 7 --restarts 3
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
    _sys.path.insert(0, str(_SRC / "models"))
    _sys.path.insert(0, str(_SRC / "evaluation"))

import argparse
import os

from experiment_utils import save_pickle
from scripts.run_multiseed_baselines import build_fold_splits_with_seed  # type: ignore
from graph_baselines import GraphConfig, train_one_split as graph_train_split
from phase4_evaluation import prepare_common_inputs


RESULTS_ROOT = _pathlib.Path(
    os.environ.get(
        "RESULTS_ROOT_DIR",
        _pathlib.Path(__file__).resolve().parent.parent.parent / "results",
    )
)
CANON_DIR = RESULTS_ROOT / "formal_doseaware_neg10_auroc" / "main_benchmark" / "fold_results"
MULTI_DIR = RESULTS_ROOT / "formal_doseaware_neg10_auroc" / "multiseed_baselines" / "fold_results"


def _pkl_path(seed: int, fold_id: int) -> _pathlib.Path:
    if seed == 42:
        return CANON_DIR / f"HGT_fold{fold_id}.pkl"
    return MULTI_DIR / f"HGT_seed{seed}_fold{fold_id}.pkl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7])
    ap.add_argument("--restarts", type=int, default=3,
                    help="Number of random-init restarts per fold; best by val AUROC wins.")
    ap.add_argument("--neg-ratio", type=int, default=10)
    args = ap.parse_args()

    print(f"[HGT-restart] seeds={args.seeds}  restarts={args.restarts}  "
          f"neg_ratio={args.neg_ratio}", flush=True)

    ds, df, _, X, labels, hp, ap_profiles, pf, lookups = prepare_common_inputs()

    for seed in args.seeds:
        splits = build_fold_splits_with_seed(df, seed)
        print(f"\n=== outer_seed={seed} ({len(splits)} folds) ===", flush=True)
        for split in splits:
            fold_id = int(split["fold"])
            out = _pkl_path(seed, fold_id)
            out.parent.mkdir(parents=True, exist_ok=True)

            best_result = None
            best_val = -1.0
            for restart in range(args.restarts):
                cfg = GraphConfig(
                    seed=seed + 1000 * restart,
                    neg_ratio=args.neg_ratio,
                )
                result = graph_train_split("HGT", ds, split, cfg, save_result=False)
                v = float(result["val_auroc"])
                print(f"  seed {seed} fold {fold_id} restart {restart}: "
                      f"val_AUROC={v:.4f}  test_AUROC={result['auroc']:.4f}",
                      flush=True)
                if v > best_val:
                    best_val = v
                    best_result = result
                    best_result["outer_seed"] = seed
                    best_result["selected_restart"] = restart
                    best_result["num_restarts"] = args.restarts

            if out.exists():
                out.unlink()
            save_pickle(best_result, out)
            assert best_result is not None
            print(f"  >> seed {seed} fold {fold_id}: SELECTED restart "
                  f"{best_result['selected_restart']}  "
                  f"val_AUROC={best_val:.4f}  "
                  f"test_AUROC={best_result['auroc']:.4f}  "
                  f"test_AUPRC={best_result['auprc']:.4f}",
                  flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
