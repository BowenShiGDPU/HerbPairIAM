"""Single-task runner for one alliance-ablation setting.

Each call trains exactly one DoseAwareIAM variant (full or
leave-one-alliance-out) on the full 10-fold pair-strat CV, with auto-skip
when fold pickles already exist. Aggregating ``alliance_ablation.csv`` is a
separate concern handled by ``aggregate_formal_alliance_ablation.py``.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/alliance_ablation \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/scripts/run_formal_alliance_ablation_job.py --setting without_AL_dose [--fold 0]
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys

from neural_models import ModelConfig, build_sample_collections
from phase4_evaluation import (
    FORMAL_ALLIANCE_ABLATIONS,
    PRIMARY_MODEL_NAME,
    _resumable_neural_cv,
    prepare_common_inputs,
    summarize_neural,
)


sys.stdout.reconfigure(line_buffering=True)


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


def _filter_splits(fold_splits, allowed: list[int] | None):
    if allowed is None:
        return fold_splits
    return [s for s in fold_splits if int(s.get("fold", s.get("seed", 0))) in allowed]


def main() -> int:
    setting_to_tags = {name: tags for name, tags in FORMAL_ALLIANCE_ABLATIONS}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=sorted(setting_to_tags.keys()))
    parser.add_argument("--neg-ratio", type=int, default=10)
    parser.add_argument(
        "--fold",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of fold ids to run (defaults to all 10).",
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
    args = parser.parse_args()

    ablation_tags = setting_to_tags[args.setting]
    tag_label = f"{PRIMARY_MODEL_NAME}__{args.setting}"

    ds, df, _, _, labels, hp, ap, pf, lookups = prepare_common_inputs()
    sample_map = build_sample_collections(
        df, lookups, hp, ap, pf, [PRIMARY_MODEL_NAME], feature_ablation=ablation_tags or None
    )
    samples = sample_map[PRIMARY_MODEL_NAME]
    cfg = frozen_cfg(args.neg_ratio)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.patience is not None:
        cfg.patience = int(args.patience)
    fold_splits = _filter_splits(ds["fold_splits"], args.fold)

    print(
        f"[alliance_ablation_job] setting={args.setting} tag={tag_label} "
        f"folds={[int(s['fold']) for s in fold_splits]} "
        f"ablation_tags={sorted(list(ablation_tags))}",
        flush=True,
    )
    results = _resumable_neural_cv(tag_label, PRIMARY_MODEL_NAME, samples, labels, fold_splits, cfg)
    summary = summarize_neural(results)
    print(tag_label, summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
