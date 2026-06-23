"""Compute-profile post-processing for the formal benchmark.

Reads existing fold artifacts from ``main_benchmark/fold_results/`` and emits
``tables/computational_profile.csv`` populated with the fields required by
EXPERIMENT_PROTOCOL.md section 11.1:

    n_parameters, training_time_per_fold_sec, inference_time_per_sample_ms,
    peak_memory_mb, hardware, framework

Tabular models (LR / RF / GB / MLP / XGBoost) currently store zeroed
``train_time_sec`` / ``inference_time_sec`` placeholders in their fold pickles.
For those models we re-fit on fold-0 once to obtain wall-clock numbers; the
result is cached to ``supplementary/compute_profile_tabular_fold0.pkl`` so the
expensive RF/GB step does not need to repeat across runs.

Usage::

    RESULTS_ROOT_DIR=results \
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \
    python -u src/compute_profile_postprocess.py [--retrain-tabular] \
                                                 [--include MODEL,...]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_utils import (
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    TABLES_DIR,
    compute_metrics,
    ensure_output_dirs,
    load_pickle,
    sanitize_name,
    save_pickle,
)


sys.stdout.reconfigure(line_buffering=True)


def _hardware_string() -> str:
    if torch.cuda.is_available():
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda:unknown"
    proc = platform.processor() or platform.machine() or "cpu"
    return f"cpu:{proc}"


def _framework_string() -> str:
    parts = [f"torch={torch.__version__}"]
    try:
        import sklearn

        parts.append(f"sklearn={sklearn.__version__}")
    except Exception:
        pass
    try:
        import xgboost

        parts.append(f"xgboost={xgboost.__version__}")
    except Exception:
        pass
    try:
        import torch_geometric

        parts.append(f"pyg={torch_geometric.__version__}")
    except Exception:
        pass
    return ", ".join(parts)


def _aggregate_fold_pkls(model_name: str, results: list[dict]) -> dict:
    train_times = np.asarray([float(r.get("train_time_sec", 0.0)) for r in results], dtype=float)
    infer_times = np.asarray([float(r.get("inference_time_sec", 0.0)) for r in results], dtype=float)
    n_tests = np.asarray([float(r.get("n_test", 1)) for r in results], dtype=float)
    mems = np.asarray([float(r.get("peak_memory_mb", float("nan"))) for r in results], dtype=float)
    if infer_times.size > 0 and float(np.nanmax(infer_times)) <= 0.0:
        per_sample_ms = np.full_like(infer_times, np.nan, dtype=float)
    else:
        per_sample_ms = np.where(n_tests > 0, infer_times / np.maximum(n_tests, 1.0) * 1000.0, np.nan)
    return {
        "model": model_name,
        "n_folds": int(len(results)),
        "training_time_per_fold_sec": float(np.nanmean(train_times)) if train_times.size else float("nan"),
        "training_time_per_fold_sec_std": float(np.nanstd(train_times, ddof=0)) if train_times.size else float("nan"),
        "inference_time_per_sample_ms": float(np.nanmean(per_sample_ms)) if per_sample_ms.size else float("nan"),
        "inference_time_per_sample_ms_std": float(np.nanstd(per_sample_ms, ddof=0)) if per_sample_ms.size else float("nan"),
        "peak_memory_mb": float(np.nanmean(mems)) if mems.size else float("nan"),
    }


def _load_fold_results() -> dict[str, list[dict]]:
    if not FOLD_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Fold results dir not found: {FOLD_RESULTS_DIR}")
    by_model: dict[str, list[dict]] = {}
    seen = set()
    for pattern in ("*_fold*.pkl", "cold_*.pkl"):
        for path in sorted(FOLD_RESULTS_DIR.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                obj = load_pickle(path)
            except Exception:
                continue
            model = str(obj.get("model", path.stem.split("_fold")[0]))
            by_model.setdefault(model, []).append(obj)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r.get("fold", r.get("cold_start_seed", 0))))
    return by_model


def _count_neural_parameters(model_name: str) -> tuple[int, str]:
    """Build the model graph once and report its trainable parameter count."""

    from neural_models import (
        DEVICE,
        ModelConfig,
        build_model,
        build_sample_collections,
        load_all,
    )

    ds, hp, ap, pf, lookups = load_all()
    df = ds["df"]
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, [model_name])
    samples = sample_map[model_name]
    sample_example = next((s for s in samples if s is not None), None)
    cfg = ModelConfig(hidden=32, dropout=0.3, lr=1e-3, epochs=1, patience=1, neg_ratio=10)
    model = build_model(model_name, cfg, sample_example=sample_example)
    n_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    device = str(next(model.parameters()).device) if n_params > 0 else str(DEVICE)
    return n_params, device


def _count_graph_parameters(model_name: str) -> tuple[int, str]:
    from graph_baselines import (
        DEVICE,
        GraphConfig,
        HGTEncoder,
        PairClassifier,
        RGCNEncoder,
        build_hgt_graph,
        build_rgcn_graph,
        load_tables,
        make_type_maps,
    )

    tables = load_tables()
    type_maps = make_type_maps(tables, include_formula=False)
    cfg = GraphConfig(hidden=32, layers=2, heads=2, dropout=0.2, lr=1e-3)
    if model_name.startswith("R-GCN"):
        graph_ctx = build_rgcn_graph(tables, type_maps)
        encoder = RGCNEncoder(
            total_nodes=graph_ctx["total_nodes"],
            num_relations=len(graph_ctx["relation_names"]),
            hidden=cfg.hidden,
            layers=cfg.layers,
            dropout=cfg.dropout,
        ).to(DEVICE)
    else:
        graph_data = build_hgt_graph(tables, type_maps)
        encoder = HGTEncoder(
            num_nodes_by_type={k: len(v) for k, v in type_maps.items()},
            metadata=graph_data.metadata(),
            hidden=cfg.hidden,
            layers=cfg.layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
        ).to(DEVICE)
    classifier = PairClassifier(cfg.hidden, cfg.dropout).to(DEVICE)
    n_params = sum(int(p.numel()) for p in list(encoder.parameters()) + list(classifier.parameters()) if p.requires_grad)
    return n_params, str(DEVICE)


def _retrain_tabular_fold0(model_names: list[str], cache_path: Path, force: bool) -> dict[str, dict]:
    if cache_path.exists() and not force:
        cached = load_pickle(cache_path)
        cached_models = set(cached.keys())
        if cached_models.issuperset(model_names):
            return cached
    from phase4_evaluation import default_best_configs, prepare_common_inputs
    from tabular_models import balance_train_indices, make_model, needs_scaler
    from sklearn.preprocessing import StandardScaler

    ds, df, feature_cols, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    split = ds["fold_splits"][0]
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    best_configs = default_best_configs()

    cached: dict[str, dict] = load_pickle(cache_path) if cache_path.exists() else {}
    for model_name in model_names:
        if model_name in cached and not force:
            continue
        params = best_configs.get(model_name)
        if not isinstance(params, dict) or "n_estimators" not in (params or {}):
            from tabular_models import MODEL_GRIDS

            params = MODEL_GRIDS[model_name][0]
        balanced = balance_train_indices(labels, train_idx, neg_ratio=10, seed=42)
        X_train = X[balanced]
        y_train = labels[balanced]
        X_test = X[test_idx]
        if needs_scaler(model_name):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        model = make_model(model_name, params)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        t0 = time.time()
        _ = model.predict_proba(X_test)
        infer_time = time.time() - t0
        n_params = _estimate_tabular_parameters(model_name, model)
        cached[model_name] = {
            "model": model_name,
            "n_parameters": int(n_params),
            "train_time_sec": float(train_time),
            "inference_time_sec_total": float(infer_time),
            "n_test": int(len(test_idx)),
            "inference_time_per_sample_ms": float(infer_time / max(len(test_idx), 1) * 1000.0),
            "params": params,
        }
        print(
            f"  retrain {model_name}: n_params~{n_params}, train={train_time:.2f}s, infer={infer_time*1000:.1f}ms total",
            flush=True,
        )
    save_pickle(cached, cache_path)
    return cached


def _estimate_tabular_parameters(model_name: str, fitted_model) -> int:
    if model_name == "LogisticRegression":
        return int(fitted_model.coef_.size + fitted_model.intercept_.size)
    if model_name == "MLP":
        return int(sum(c.size for c in fitted_model.coefs_) + sum(b.size for b in fitted_model.intercepts_))
    if model_name == "RandomForest":
        n_nodes = int(sum(int(est.tree_.node_count) for est in fitted_model.estimators_))
        return n_nodes
    if model_name == "GradientBoosting":
        try:
            n_nodes = int(sum(int(est[0].tree_.node_count) for est in fitted_model.estimators_))
        except Exception:
            n_nodes = int(getattr(fitted_model, "n_estimators", 0))
        return n_nodes
    if model_name == "XGBoost":
        try:
            booster = fitted_model.get_booster()
            df_trees = booster.trees_to_dataframe()
            return int(len(df_trees))
        except Exception:
            return int(getattr(fitted_model, "n_estimators", 0))
    return 0


def _resolve_param_count(model_name: str, tabular_cache: dict | None) -> tuple[int, str]:
    if model_name in {"LogisticRegression", "RandomForest", "GradientBoosting", "MLP", "XGBoost"}:
        if tabular_cache and model_name in tabular_cache:
            return int(tabular_cache[model_name]["n_parameters"]), "cpu"
        return 0, "cpu"
    if model_name in {"R-GCN", "HGT"}:
        return _count_graph_parameters(model_name)
    try:
        return _count_neural_parameters(model_name)
    except Exception as exc:
        print(f"  WARN: failed to count parameters for {model_name}: {exc}", flush=True)
        return 0, str(_hardware_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain-tabular",
        action="store_true",
        help="Force re-fit of tabular models on fold 0 to refresh wall-clock cache.",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Comma-separated subset of models. Default: every model with fold pickles.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    fold_results = _load_fold_results()
    if not fold_results:
        print("No fold results found, nothing to do.")
        return 1
    target_models = set(fold_results)
    if args.include:
        target_models &= {m.strip() for m in args.include.split(",") if m.strip()}
    if not target_models:
        print("Selected --include filter has no overlap with available fold results.")
        return 1

    tabular_models = [m for m in target_models if m in {"LogisticRegression", "RandomForest", "GradientBoosting", "MLP", "XGBoost"}]
    tabular_cache: dict[str, dict] = {}
    if tabular_models:
        cache_path = SUPP_DIR / "compute_profile_tabular_fold0.pkl"
        print(f"Refreshing tabular timings into {cache_path} for: {tabular_models}", flush=True)
        tabular_cache = _retrain_tabular_fold0(tabular_models, cache_path, force=args.retrain_tabular)

    hardware = _hardware_string()
    framework = _framework_string()

    rows = []
    for model_name in sorted(target_models):
        results = fold_results[model_name]
        agg = _aggregate_fold_pkls(model_name, results)
        n_params, device = _resolve_param_count(model_name, tabular_cache)
        if model_name in tabular_cache:
            cache_entry = tabular_cache[model_name]
            agg["training_time_per_fold_sec"] = float(cache_entry["train_time_sec"])
            agg["training_time_per_fold_sec_std"] = float("nan")
            agg["inference_time_per_sample_ms"] = float(cache_entry["inference_time_per_sample_ms"])
            agg["inference_time_per_sample_ms_std"] = float("nan")
        rows.append(
            {
                **agg,
                "n_parameters": int(n_params),
                "hardware": hardware,
                "framework": framework,
                "device_used": device,
            }
        )

    new_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "computational_profile.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[~existing["model"].isin(new_df["model"])]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values("model").reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(merged)} rows; updated {len(new_df)} models).")
    print(merged.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
