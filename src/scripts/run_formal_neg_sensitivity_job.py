"""Single-task runner for one (model, neg_ratio) cell of EXPERIMENT_PROTOCOL s9.1.

Each call trains exactly one of (model x neg_ratio) on the full 10-fold pair-strat
CV with auto-skip of fold pickles already saved at::

    <stage>/fold_results/<sanitize(model)>_neg<R>_fold<K>.pkl

The aggregating ``supplementary/neg_sampling_sensitivity.csv`` table is built
by ``aggregate_formal_neg_sensitivity.py`` after all (model, neg_ratio) jobs
finish.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/neg_sampling_sensitivity \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/run_formal_neg_sensitivity_job.py \\
        --model DoseAwareIAM --neg-ratio 1
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys

from neural_models import ModelConfig, build_sample_collections, summarize_results as summarize_neural
from tabular_models import search_params, summarize_results as summarize_tabular
from graph_baselines import GraphConfig, summarize_results as summarize_graph
from phase4_evaluation import (
    NEG_RATIO_GRID,
    NEG_SENSITIVITY_GRAPH,
    NEG_SENSITIVITY_NEURAL,
    NEG_SENSITIVITY_TABULAR,
    PRIMARY_MODEL_NAME,
    _resumable_neg_graph_cv,
    _resumable_neg_neural_cv,
    _resumable_neg_tabular_cv,
    graph_candidate_configs,
    prepare_common_inputs,
)


sys.stdout.reconfigure(line_buffering=True)


SUPPORTED_MODELS = list({*NEG_SENSITIVITY_TABULAR, *NEG_SENSITIVITY_NEURAL, *NEG_SENSITIVITY_GRAPH})


def frozen_neural_cfg(neg_ratio: int, epochs: int | None, patience: int | None) -> ModelConfig:
    cfg = ModelConfig(
        hidden=32,
        dropout=0.3,
        lr=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        neg_ratio=neg_ratio,
        eval_every=2,
    )
    if epochs is not None:
        cfg.epochs = int(epochs)
    if patience is not None:
        cfg.patience = int(patience)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--neg-ratio", type=int, required=True, choices=NEG_RATIO_GRID)
    parser.add_argument(
        "--fold",
        type=int,
        nargs="+",
        default=None,
        help="Optional fold subset for dry runs (defaults to all 10).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override neural epochs (dry-run).")
    parser.add_argument("--patience", type=int, default=None, help="Override neural patience (dry-run).")
    args = parser.parse_args()

    ds, df, _, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    fold_splits = ds["fold_splits"]
    if args.fold is not None:
        fold_splits = [s for s in fold_splits if int(s.get("fold", s.get("seed", 0))) in args.fold]
    print(
        f"[neg_sens_job] model={args.model} neg_ratio=1:{args.neg_ratio} folds={[int(s['fold']) for s in fold_splits]}",
        flush=True,
    )

    if args.model in NEG_SENSITIVITY_TABULAR:
        params = search_params(args.model, X, labels, ds["fold_splits"][0], neg_ratio=10)
        results = _resumable_neg_tabular_cv(args.model, X, labels, fold_splits, args.neg_ratio, params)
        summary = summarize_tabular(results)
    elif args.model in NEG_SENSITIVITY_NEURAL:
        sample_map = build_sample_collections(df, lookups, hp, ap, pf, [args.model])
        cfg = frozen_neural_cfg(args.neg_ratio, args.epochs, args.patience)
        results = _resumable_neg_neural_cv(args.model, sample_map[args.model], labels, fold_splits, args.neg_ratio, cfg)
        summary = summarize_neural(results)
    elif args.model in NEG_SENSITIVITY_GRAPH:
        graph_cfg = graph_candidate_configs()[0]
        if args.epochs is not None:
            graph_cfg = GraphConfig(**{**graph_cfg.__dict__, "epochs": int(args.epochs)})
        if args.patience is not None:
            graph_cfg = GraphConfig(**{**graph_cfg.__dict__, "patience": int(args.patience)})
        results = _resumable_neg_graph_cv(args.model, ds, fold_splits, args.neg_ratio, graph_cfg)
        summary = summarize_graph(results)
    else:
        raise SystemExit(f"Unsupported model {args.model}")

    print(f"{args.model} neg=1:{args.neg_ratio} ({len(results)} folds):", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
