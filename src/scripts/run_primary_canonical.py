"""Canonical training run for the primary model.

Produces the ``HerbPairIAM_fold*.pkl`` and ``HerbPairIAM_fold*.pt`` files
that the main benchmark table, cluster bootstrap CI and Phase 5
interpretability all depend on. Uses the frozen protocol config and the
strict-determinism hook enabled by ``enable_strict_determinism``.

Why a dedicated script?
-----------------------
Head-to-head dose diagnostics (``run_dose_head2head.py``) trained the
primary model with ``save_model=False`` to stay lightweight. For
publication we need the state dicts (``.pt``) so that
``phase5_interpretability`` can reload each fold's best weights and
recover attention patterns on its own OOF test samples. We therefore
rerun those 10 folds once, canonically, under strict determinism.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/scripts/run_primary_canonical.py \\
        [--fold 0 1 2 ...] [--model HerbPairIAM] [--no-skip]

By default the script skips any fold that already has a ``.pkl`` under
``EXPERIMENT_SUBDIR/fold_results/``; use ``--no-skip`` to retrain.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys

from experiment_utils import FOLD_RESULTS_DIR, fold_result_path, save_pickle
from neural_models import ModelConfig, build_sample_collections, summarize_results, train_one_split
from phase4_evaluation import PRIMARY_MODEL_NAME, prepare_common_inputs


sys.stdout.reconfigure(line_buffering=True)


def frozen_cfg() -> ModelConfig:
    """Canonical training configuration — see PRIMARY_MODEL.md."""
    return ModelConfig(
        hidden=32,
        dropout=0.3,
        lr=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        neg_ratio=10,
        eval_every=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRIMARY_MODEL_NAME,
                        help=f"Neural model to train (default: {PRIMARY_MODEL_NAME})")
    parser.add_argument("--fold", type=int, nargs="+", default=None,
                        help="Subset of fold ids (default: all 10)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Retrain even when the fold pkl already exists")
    args = parser.parse_args()

    ds, df, _, _, labels, hp, ap, pf, lookups = prepare_common_inputs()
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, [args.model])
    samples = sample_map[args.model]

    cfg = frozen_cfg()
    fold_splits = ds["fold_splits"]
    if args.fold is not None:
        allowed = set(int(f) for f in args.fold)
        fold_splits = [s for s in fold_splits if int(s["fold"]) in allowed]

    print(
        f"[primary_canonical] model={args.model} folds={[int(s['fold']) for s in fold_splits]} "
        f"cfg={cfg.__dict__}",
        flush=True,
    )

    results = []
    for split in fold_splits:
        fold_id = int(split["fold"])
        target_pkl = fold_result_path(args.model, fold_id)
        if target_pkl.exists() and not args.no_skip:
            print(f"  [skip] {target_pkl.name} already exists", flush=True)
            import pickle
            with open(target_pkl, "rb") as f:
                results.append(pickle.load(f))
            continue
        print(f"  training fold={fold_id} ...", flush=True)
        # train_one_split defaults to save_model=True — state dict will be
        # persisted under MODELS_DIR/{model}_fold{fold}.pt.
        result = train_one_split(args.model, samples, labels, split, cfg)
        save_pickle(result, target_pkl)
        print(
            f"  fold {fold_id}: AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}  "
            f"state={result.get('model_state_path')}",
            flush=True,
        )
        results.append(result)

    summary = summarize_results(results)
    print(
        f"[primary_canonical] {args.model} over {len(results)} folds: "
        f"AUROC={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f}  "
        f"AUPRC={summary['auprc_mean']:.4f}±{summary['auprc_std']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
