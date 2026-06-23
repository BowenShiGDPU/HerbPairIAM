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
    FORMAL_FEATURE_ABLATIONS,
    PRIMARY_MODEL_NAME,
    _resumable_neural_cv,
    _resumable_tabular_cv,
    build_feature_groups,
    prepare_common_inputs,
    summarize_neural,
    summarize_tabular,
)
from tabular_models import search_params


sys.stdout.reconfigure(line_buffering=True)


def frozen_cfg() -> ModelConfig:
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


def main():
    parser = argparse.ArgumentParser()
    # Any neural primary (HerbPairIAM / DoseAwareIAM historically) plus the
    # tabular XGBoost baseline. DoseAwareIAM is kept as a valid choice so old
    # feature_ablation fold pickles can still be regenerated if needed.
    parser.add_argument("--model", required=True, choices=["HerbPairIAM", "DoseAwareIAM", "XGBoost"])
    parser.add_argument("--setting", required=True)
    parser.add_argument("--neg-ratio", type=int, default=10)
    args = parser.parse_args()

    ds, df, feature_cols, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    feature_groups = build_feature_groups(ds["feature_cols"])
    setting_to_ablation = {name: ablation for name, ablation in FORMAL_FEATURE_ABLATIONS}

    if args.model == "XGBoost":
        if args.setting not in feature_groups:
            raise ValueError(f"Unknown XGBoost feature setting: {args.setting}")
        tag = f"XGBoost__{args.setting}"
        params = search_params("XGBoost", X, labels, ds["fold_splits"][0], neg_ratio=args.neg_ratio)
        results = _resumable_tabular_cv(tag, "XGBoost", X, labels, ds["fold_splits"], params, args.neg_ratio, feature_groups[args.setting])
        summary = summarize_tabular(results)
        print(tag, summary, flush=True)
        return

    if args.setting not in setting_to_ablation:
        raise ValueError(f"Unknown feature setting: {args.setting}")
    tag = f"{args.model}__{args.setting}"
    sample_map = build_sample_collections(
        df,
        lookups,
        hp,
        ap,
        pf,
        [args.model],
        feature_ablation=setting_to_ablation[args.setting],
    )
    cfg = frozen_cfg()
    cfg.neg_ratio = args.neg_ratio
    results = _resumable_neural_cv(tag, args.model, sample_map[args.model], labels, ds["fold_splits"], cfg)
    summary = summarize_neural(results)
    print(tag, summary, flush=True)


if __name__ == "__main__":
    main()
