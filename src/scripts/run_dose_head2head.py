"""Step-1 head-to-head: did dose actually help? — multi-seed, multi-model.

Background
----------
Across the formal experiments we see that the four "dose-removed" candidates
beat full ``DoseAwareIAM`` on the same 10-fold split (``without_AL_dose`` and
``without_dose`` reach 0.82+ AUROC vs 0.81 for ``DoseAwareIAM``), yet the
strict no-dose ``InteractionAwareSetModel`` (V0) does NOT — it sits at 0.8139,
between them. The likely confound is **model capacity**: DoseAwareIAM has
``dose_enc`` + an extra ``hidden`` slot in ``pred`` (~6.7k extra params,
+66% over IAM). When dose values are zeroed (``without_dose``), those extra
parameters still act as a free regularization channel. So the head-to-head
must include a capacity-matched IAM to disambiguate.

Models compared (default)
-------------------------
* V0  ``InteractionAwareSetModel`` — strict no-dose IAM, hidden=32, ~10k params
* V0w ``IAM_Wide``                 — IAM with hidden=44 (~17k params), capacity-
                                     matched to DoseAwareIAM but no dose
* V4z ``DoseAware_ZeroDose``       — full DoseAwareIAM architecture but every
                                     dose-derived sample field is zero-filled
                                     via feature_ablation={"AL_dose"}
* V4  ``DoseAwareIAM``             — full model with real dose values

Decision logic
--------------
* V0 ≈ V0w ≈ V4z ≈ V4   — every model is equivalent → choose the simplest (V0).
* V0 < V0w ≈ V4z ≈ V4   — capacity matters, dose does not → V0w or V4z.
* V0 ≈ V0w < V4z ≈ V4   — DoseAware structure regularizes regardless of dose
                          values → V4z.
* V4 strictly best       — dose really helps → keep V4.

Usage
-----
::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/dose_head2head \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/scripts/run_dose_head2head.py \\
        --seeds 42 13 7 \\
        --models InteractionAwareSetModel IAM_Wide DoseAware_ZeroDose DoseAwareIAM \\
        --neg-ratio 10

Each (seed, model, fold) triple is cached as a pickle, so the job is resumable.
After all combinations finish, ``dose_head2head_summary.csv`` and
``dose_head2head_pooled.csv`` are written under ``EXPERIMENT_SUBDIR/tables/``.
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
from evaluation.experiment_utils import (
    FOLD_RESULTS_DIR,
    TABLES_DIR,
    load_pickle,
    sanitize_name,
    save_pickle,
)
from models.neural_models import (
    ModelConfig,
    build_sample_collections,
    summarize_results,
    train_one_split,
)
from evaluation.phase4_evaluation import prepare_common_inputs


sys.stdout.reconfigure(line_buffering=True)


DEFAULT_MODELS = [
    "InteractionAwareSetModel",  # V0  — strict no dose, hidden=32
    "IAM_Wide",                   # V0w — strict no dose, hidden=44 (capacity-matched)
    "HerbPairIAM",                # V4z — PRIMARY MODEL: full DoseAware arch, dose zeroed
    "DoseAwareIAM",               # V4  — DoseAware fed real dose (ablation baseline)
]
DEFAULT_SEEDS = [42, 13, 7]
# Models whose samples need zero-dose fields. As of the HerbPairIAM naming this
# is enforced automatically by ``neural_models.model_intrinsic_ablation``, so
# the set is kept here only for backward-compatible historical references.
ZERO_DOSE_MODELS = frozenset({"HerbPairIAM", "DoseAware_ZeroDose"})


def build_fold_splits_with_seed(df: pd.DataFrame, outer_seed: int) -> list[dict]:
    """Recreate the same split topology used by ``build_fold_splits`` but with a
    user-controlled outer CV seed. Inner validation seeding mirrors the
    production logic (``outer_seed + fold_id``)."""

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


def _h2h_fold_path(tag: str, fold_id: int) -> _pathlib.Path:
    return FOLD_RESULTS_DIR / f"{sanitize_name(tag)}_fold{fold_id}.pkl"


def _resumable_one_seed(
    seed: int,
    model_name: str,
    samples,
    labels,
    fold_splits: list[dict],
    cfg: ModelConfig,
) -> list[dict]:
    tag = f"H2H_seed{seed}_{model_name}"
    results = []
    for split in fold_splits:
        fold_id = int(split["fold"])
        path = _h2h_fold_path(tag, fold_id)
        if path.exists():
            results.append(load_pickle(path))
            continue
        fold_result = train_one_split(model_name, samples, labels, split, cfg, save_model=False)
        fold_result["ablation_tag"] = tag
        fold_result["outer_seed"] = seed
        fold_result["model_name"] = model_name
        save_pickle(fold_result, path)
        results.append(fold_result)
    return sorted(results, key=lambda item: int(item["fold"]))


def frozen_cfg(neg_ratio: int) -> ModelConfig:
    return ModelConfig(
        hidden=32,
        dropout=0.3,
        lr=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        neg_ratio=neg_ratio,
        eval_every=2,
    )


def _filter_folds(splits: list[dict], allowed: list[int] | None) -> list[dict]:
    if allowed is None:
        return splits
    allowed_set = set(int(f) for f in allowed)
    return [s for s in splits if int(s["fold"]) in allowed_set]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Outer CV seeds. Default: 42 13 7.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to include in the head-to-head.",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=10,
        help="Training negative sampling ratio (matches main protocol).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of fold ids (defaults to all 10).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs (used for fast dry runs).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early-stopping patience (used for fast dry runs).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing dose_head2head_summary.csv (useful when running shards).",
    )
    args = parser.parse_args()

    ds, df, _, _, labels, hp, ap, pf, lookups = prepare_common_inputs()

    # build_sample_collections transparently applies model_intrinsic_ablation
    # so HerbPairIAM / DoseAware_ZeroDose automatically get zero-dose samples
    # while other models (IAM, IAM_Wide, DoseAwareIAM) get normal samples.
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, args.models)

    cfg = frozen_cfg(args.neg_ratio)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.patience is not None:
        cfg.patience = int(args.patience)

    rows = []
    for seed in args.seeds:
        fold_splits = _filter_folds(build_fold_splits_with_seed(df, seed), args.fold)
        fold_ids = [int(s["fold"]) for s in fold_splits]
        for model_name in args.models:
            print(
                f"[dose_head2head] seed={seed} model={model_name} folds={fold_ids}",
                flush=True,
            )
            samples = sample_map[model_name]
            results = _resumable_one_seed(seed, model_name, samples, labels, fold_splits, cfg)
            summary = summarize_results(results)
            print(
                f"[dose_head2head] DONE seed={seed} model={model_name} "
                f"AUROC={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f} "
                f"AUPRC={summary['auprc_mean']:.4f}±{summary['auprc_std']:.4f}",
                flush=True,
            )
            rows.append(
                {
                    "outer_seed": seed,
                    "model": model_name,
                    "n_folds": len(results),
                    "AUROC_mean": summary["auroc_mean"],
                    "AUROC_std": summary["auroc_std"],
                    "AUPRC_mean": summary["auprc_mean"],
                    "AUPRC_std": summary["auprc_std"],
                    "F1_mean": summary["f1_mean"],
                    "MCC_mean": summary["mcc_mean"],
                }
            )

    if not args.no_summary and rows:
        summary_df = pd.DataFrame(rows)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TABLES_DIR / "dose_head2head_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"[dose_head2head] wrote {out_path}", flush=True)

        agg = (
            summary_df.groupby("model")
            .agg(
                seeds=("outer_seed", lambda x: sorted(set(int(v) for v in x))),
                AUROC_pooled_mean=("AUROC_mean", "mean"),
                AUROC_seed_std=("AUROC_mean", "std"),
                AUPRC_pooled_mean=("AUPRC_mean", "mean"),
                AUPRC_seed_std=("AUPRC_mean", "std"),
                n_seed_x_fold=("n_folds", "sum"),
            )
            .reset_index()
        )
        agg_path = TABLES_DIR / "dose_head2head_pooled.csv"
        agg.to_csv(agg_path, index=False)
        print(f"[dose_head2head] wrote {agg_path}", flush=True)
        print(agg.to_string(index=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
