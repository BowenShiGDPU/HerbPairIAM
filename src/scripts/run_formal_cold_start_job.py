"""Single-task runner for one (split_type, seed, model) cold-start cell.

Each call trains exactly one cold-start cell (Formula CS or ADR CS at one of
the 5 seeds) of one model. The fold pickle is dropped under::

    <stage>/fold_results/cold_<split_type>_<seed>_<sanitize(model)>.pkl

After all jobs finish, ``aggregate_formal_cold_start.py`` builds
``tables/cold_start.csv`` and ``supplementary/cold_start_progress.csv``.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/cold_start \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/run_formal_cold_start_job.py \\
        --split-type Formula --seed 42 --model DoseAwareIAM
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
from tabular_models import search_params
from graph_baselines import GraphConfig
from phase4_evaluation import (
    COLD_START_GRAPH_MODELS,
    COLD_START_NEURAL_MODELS,
    COLD_START_SPLIT_TYPES,
    COLD_START_TABULAR_MODELS,
    PRIMARY_MODEL_NAME,
    _resumable_cold_graph,
    _resumable_cold_neural,
    _resumable_cold_tabular,
    default_best_configs,
    graph_candidate_configs,
    prepare_common_inputs,
    resolve_neural_config,
)


sys.stdout.reconfigure(line_buffering=True)


SUPPORTED_MODELS = list({*COLD_START_TABULAR_MODELS, *COLD_START_NEURAL_MODELS, *COLD_START_GRAPH_MODELS})


def _select_split(splits: list[dict], seed: int) -> dict | None:
    for s in splits:
        if int(s.get("seed", -1)) == seed:
            return s
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-type", required=True, choices=COLD_START_SPLIT_TYPES)
    parser.add_argument("--seed", type=int, required=True, help="Cold-start seed (42, 88, 999, 777 or 666).")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--neg-ratio", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None, help="Override neural/graph epochs (dry-run).")
    parser.add_argument("--patience", type=int, default=None, help="Override neural/graph patience (dry-run).")
    args = parser.parse_args()

    ds, df, _, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    splits_lookup = {
        "Formula": ds.get("formula_cs_splits", []),
        "ADR": ds.get("adr_cs_splits", []),
    }
    split = _select_split(splits_lookup[args.split_type], args.seed)
    if split is None:
        raise SystemExit(f"No {args.split_type} cold-start split found for seed={args.seed}")

    print(
        f"[cold_start_job] split_type={args.split_type} seed={args.seed} model={args.model} primary={PRIMARY_MODEL_NAME}",
        flush=True,
    )

    if args.model in COLD_START_TABULAR_MODELS:
        params = default_best_configs().get(args.model) or search_params(args.model, X, labels, ds["fold_splits"][0], neg_ratio=args.neg_ratio)
        results = _resumable_cold_tabular(args.model, X, labels, [split], args.split_type, params, args.neg_ratio)
    elif args.model in COLD_START_NEURAL_MODELS:
        sample_map = build_sample_collections(df, lookups, hp, ap, pf, [args.model])
        cfg = resolve_neural_config(args.model, sample_map, labels, ds["fold_splits"], default_best_configs(), neg_ratio=args.neg_ratio)
        if args.epochs is not None:
            cfg.epochs = int(args.epochs)
        if args.patience is not None:
            cfg.patience = int(args.patience)
        results = _resumable_cold_neural(args.model, sample_map[args.model], labels, [split], args.split_type, cfg, args.neg_ratio)
    elif args.model in COLD_START_GRAPH_MODELS:
        graph_cfg = graph_candidate_configs()[0]
        if args.epochs is not None:
            graph_cfg = GraphConfig(**{**graph_cfg.__dict__, "epochs": int(args.epochs)})
        if args.patience is not None:
            graph_cfg = GraphConfig(**{**graph_cfg.__dict__, "patience": int(args.patience)})
        results = _resumable_cold_graph(args.model, ds, [split], args.split_type, graph_cfg, args.neg_ratio)
    else:
        raise SystemExit(f"Unsupported model {args.model}")

    if results:
        r = results[0]
        print(
            f"  AUROC={float(r['auroc']):.4f}  AUPRC={float(r['auprc']):.4f}  "
            f"F1={float(r['f1']):.4f}  MCC={float(r['mcc']):.4f}  "
            f"n_test={int(r.get('n_test', len(r.get('y_true', []))))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
