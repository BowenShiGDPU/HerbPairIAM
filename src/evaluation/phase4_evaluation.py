"""
Phase 4: Experiment runner for bake-off and benchmark tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import (
    FULL_EXPERIMENT_DIR,
    SUPP_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    VAL_SELECTION_METRIC,
    calibration_table,
    compute_metrics,
    compute_pooled_predictions,
    delong_test,
    ensure_output_dirs,
    format_metric,
    holm_bonferroni,
    load_pickle,
    paired_ttest_matrix,
    save_pickle,
    write_run_manifest,
)
from neural_models import (
    ModelConfig,
    build_sample_collections,
    load_all,
    train_one_split as train_neural_split,
    run_cv as run_neural_cv,
    search_config,
    summarize_results as summarize_neural,
)
from tabular_models import fit_predict_split, run_cv as run_tabular_cv, search_params, summarize_results as summarize_tabular
from graph_baselines import GraphConfig, run_cv as run_graph_cv, summarize_results as summarize_graph


sys.stdout.reconfigure(line_buffering=True)
ROOT_OUT = Path(__file__).resolve().parent.parent.parent / "outputs"
EVAL_SUMMARY_PATH = FULL_EXPERIMENT_DIR / "evaluation_results.csv"
CHAMPION_PATH = SUPP_DIR / "champion_selection.pkl"


TABULAR_MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP", "XGBoost"]
NEURAL_BAKEOFF_MODELS = ["AttentionPool_NoMP", "HerbInteractionGraph", "InteractionAwareSetModel"]
NEURAL_ABLATION_MODELS = ["SumPool", "MeanPool", "NoADRConditioning", "DotScorer", "TwoLayerMP"]
GRAPH_MAIN_MODELS = ["R-GCN", "HGT"]
# Primary model of the project. Architecture equals DoseAwareIAM but samples
# are built with AL_dose zero-fill (enforced by
# ``neural_models.model_intrinsic_ablation``). See ``models/herb_pair_iam.py``
# and ``PRIMARY_MODEL.md`` for the justification and the head-to-head
# experiment that established this choice.
PRIMARY_MODEL_NAME = "HerbPairIAM"
# Kept as a named ablation baseline ("DoseAware fed with real dose"), not as
# the primary model. Scripts that explicitly want to compare against the
# "with-real-dose" variant should use this constant.
PRIMARY_WITH_DOSE_BASELINE = "DoseAwareIAM"
QUICK_DEFAULT_MODELS = ["XGBoost", PRIMARY_MODEL_NAME, "InteractionAwareSetModel", "KGEmbedIAM", "IngredientLiteIAM"]
# Structure-level ablation set. HerbPairIAM is the primary; DoseAwareIAM is
# kept as the "with real dose" baseline; the other DoseAware* entries remain
# as finer-grained structure ablations on the shared DoseAware architecture.
DOSEAWARE_STRUCTURE_MODELS = [
    "HerbPairIAM",
    # With-real-dose variant — demonstrates that feeding real dose values
    # hurts performance (paper's "dose ablation" claim in Table 1).
    "DoseAwareIAM",
    # No auxiliary branch — simplest IAM.
    "InteractionAwareSetModel",
    # Capacity-matched control: IAM with hidden=44, ~17k params, to show
    # HerbPairIAM's gain is not explained by capacity. Fold pickles are
    # hard-linked from the 3-seed head-to-head experiment (seed=42).
    "IAM_Wide",
    "DoseAwareNoDoseGate",
    "DoseAwareHerbOnly",
    "DoseAwarePairOnly",
    "DoseAwareNoADRConditioning",
    "DoseAwareMeanPool",
]
FORMAL_FEATURE_ABLATIONS = [
    ("full", None),
    ("without_dose", "without_dose"),
    ("without_pathway", "without_pathway"),
    ("without_tissue", "without_tissue"),
    ("without_ppi", "without_ppi"),
    ("without_convergence", "without_convergence"),
    ("without_complementarity", "without_complementarity"),
    ("without_union_coverage", "without_union_coverage"),
    # Diagnostic: zero every pair feature value (all 11 channels in
    # PAIR_FEATURE_KEYS) while keeping the pair branch's MLP and attention.
    # Tells us whether the pair branch's +0.12 AUROC over HerbOnly comes from
    # the raw feature values or from attention over herb co-occurrence.
    ("pair_zero_diagnostic", frozenset({"AL_pair_all"})),
]


# Alliance-style ablations (per docs/AllianceAblationPlan): leave-one-alliance-out
# leaves all other alliances intact. Each tuple is (setting_name, frozenset of
# ablation tags consumed by neural_models.precompute_samples /
# _ablate_pair_feature_dict). Setting "full" = no tags.
FORMAL_ALLIANCE_ABLATIONS = [
    ("full", frozenset()),
    ("without_AL_dose", frozenset({"AL_dose"})),
    ("without_AL_pair_direct", frozenset({"AL_pair_direct"})),
    ("without_AL_pair_multiomics", frozenset({"AL_pair_multiomics"})),
    ("without_AL_individual", frozenset({"AL_individual"})),
]


def build_feature_groups(feature_cols):
    indiv = [
        c
        for c in feature_cols
        if any(k in c for k in ["indiv", "n_herbs", "frac_herbs", "n_adr_targets", "max_dose_overlap"])
    ]
    interaction = [c for c in feature_cols if c not in indiv and c not in ["total_dose", "dose_std", "dose_gini", "n_formula_pairs"]]
    no_dose = [c for c in feature_cols if "dose_convergence" not in c and c not in ["total_dose", "dose_std", "dose_gini"]]
    no_pw = [c for c in feature_cols if "pw_" not in c]
    no_tis = [c for c in feature_cols if "tissue" not in c]
    no_ppi = [c for c in feature_cols if "ppi" not in c]
    no_conv = [c for c in feature_cols if "convergence" not in c or "complementarity" in c]
    no_comp = [c for c in feature_cols if "complementarity" not in c]
    no_union = [c for c in feature_cols if "union_coverage" not in c]
    as_idx = lambda cols: [feature_cols.index(c) for c in cols if c in feature_cols]
    return {
        "full": None,
        "individual_only": as_idx(indiv),
        "interaction_only": as_idx(interaction),
        "without_dose": as_idx(no_dose),
        "without_pathway": as_idx(no_pw),
        "without_tissue": as_idx(no_tis),
        "without_ppi": as_idx(no_ppi),
        "without_convergence": as_idx(no_conv),
        "without_complementarity": as_idx(no_comp),
        "without_union_coverage": as_idx(no_union),
    }


def prepare_common_inputs():
    ensure_output_dirs()
    # Emit a run manifest (git, dataset sha256, env, versions, host) into the
    # current EXPERIMENT_SUBDIR before any training or aggregation happens.
    # Existing manifests are preserved in run_manifest_history.jsonl so the
    # provenance of earlier fold pickles is not silently overwritten.
    try:
        write_run_manifest()
    except Exception as exc:  # pragma: no cover — manifests must not break training
        print(f"[warn] write_run_manifest failed: {exc}", flush=True)
    ds, hp, ap, pf, lookups = load_all()
    df = ds["df"]
    labels = df["label"].values.astype(int)
    feature_cols = ds["feature_cols"]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32), 0.0)
    return ds, df, feature_cols, X, labels, hp, ap, pf, lookups


def candidate_configs():
    return [
        ModelConfig(hidden=32, dropout=0.2, lr=5e-3, epochs=60, patience=12, eval_every=2),
        ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=80, patience=16, eval_every=2),
        ModelConfig(hidden=64, dropout=0.3, lr=1e-3, epochs=80, patience=16, eval_every=2),
    ]


def default_best_configs():
    shared = ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=40, patience=10, eval_every=2).__dict__.copy()
    light = ModelConfig(hidden=32, dropout=0.2, lr=5e-3, epochs=40, patience=10, eval_every=2).__dict__.copy()
    formal_doseaware = ModelConfig(hidden=32, dropout=0.3, lr=1e-3, epochs=100, patience=10, eval_every=2).__dict__.copy()
    return {
        "XGBoost": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
        "InteractionAwareSetModel": shared.copy(),
        "DoseAwareIAM": formal_doseaware.copy(),
        "HerbPairIAM": formal_doseaware.copy(),
        "KGEmbedIAM": shared.copy(),
        "IngredientLiteIAM": shared.copy(),
        "HerbInteractionGraph": shared.copy(),
        "AttentionPool_NoMP": light.copy(),
    }


def quick_best_configs():
    best = default_best_configs()
    shared = ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=24, patience=6, eval_every=2).__dict__.copy()
    light = ModelConfig(hidden=32, dropout=0.2, lr=5e-3, epochs=24, patience=6, eval_every=2).__dict__.copy()
    for model_name in ["InteractionAwareSetModel", "DoseAwareIAM", "HerbPairIAM", "KGEmbedIAM", "IngredientLiteIAM", "HerbInteractionGraph"]:
        best[model_name] = shared.copy()
    best["AttentionPool_NoMP"] = light.copy()
    return best


def resolve_best_configs(champion_artifact: dict | None):
    best = default_best_configs()
    if champion_artifact:
        best.update(champion_artifact.get("best_configs", {}))
    return best


def parse_model_allowlist(raw: str | None, default_models: list[str]) -> list[str]:
    if not raw:
        return default_models[:]
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models or default_models[:]


def parse_int_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(v) for v in values] if values else None


def split_selected_models(model_names: list[str]):
    tabular = [m for m in model_names if m in TABULAR_MODELS]
    neural = [m for m in model_names if m not in TABULAR_MODELS and m not in GRAPH_MAIN_MODELS]
    graph = [m for m in model_names if m in GRAPH_MAIN_MODELS]
    return tabular, neural, graph


def resolve_neural_config(model_name, sample_map, labels, fold_splits, best_configs, neg_ratio: int):
    cfg_entry = best_configs.get(model_name)
    if cfg_entry is None:
        cfg = search_config(model_name, sample_map[model_name], labels, fold_splits[0], candidate_configs())
        best_configs[model_name] = cfg.__dict__.copy()
    else:
        cfg = ModelConfig(**cfg_entry) if isinstance(cfg_entry, dict) else cfg_entry
    cfg.neg_ratio = neg_ratio
    return cfg


def _manual_neural_cv(model_name: str, samples, labels, fold_splits, cfg: ModelConfig):
    results = []
    for split in fold_splits:
        results.append(train_neural_split(model_name, samples, labels, split, cfg, save_model=False))
    return results


def _ablation_fold_path(tag: str, fold_id: int):
    from experiment_utils import FOLD_RESULTS_DIR, sanitize_name

    return FOLD_RESULTS_DIR / f"{sanitize_name(tag)}_fold{fold_id}.pkl"


def _write_ablation_summary(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _resumable_neural_cv(tag: str, model_name: str, samples, labels, fold_splits, cfg: ModelConfig):
    from experiment_utils import load_pickle, save_pickle

    results = []
    for split in fold_splits:
        fold_id = int(split.get("fold", split.get("seed", 0)))
        path = _ablation_fold_path(tag, fold_id)
        if path.exists():
            results.append(load_pickle(path))
            continue
        fold_result = train_neural_split(model_name, samples, labels, split, cfg, save_model=False)
        fold_result["ablation_tag"] = tag
        save_pickle(fold_result, path)
        results.append(fold_result)
    return sorted(results, key=lambda item: int(item["fold"]))


def _resumable_tabular_cv(tag: str, model_name: str, X, y, fold_splits, params, neg_ratio: int, feature_idx):
    from experiment_utils import load_pickle, save_pickle

    results = []
    for split in fold_splits:
        fold_id = int(split.get("fold", split.get("seed", 0)))
        path = _ablation_fold_path(tag, fold_id)
        if path.exists():
            results.append(load_pickle(path))
            continue
        result = fit_predict_split(model_name, X, y, split, params, neg_ratio=neg_ratio, seed=42, feature_idx=feature_idx)
        result["ablation_tag"] = tag
        save_pickle(result, path)
        results.append(result)
    return sorted(results, key=lambda item: int(item["fold"]))


def run_formal_feature_ablation(ds, df, X, y, hp, ap, pf, lookups, champion_config, neg_ratio: int = 10):
    print("\n== Formal Feature-level Ablations ==")
    feature_groups = build_feature_groups(ds["feature_cols"])
    rows = []
    summary_path = TABLES_DIR / "feature_ablation.csv"

    xgb_params = search_params("XGBoost", X, y, ds["fold_splits"][0], neg_ratio=neg_ratio)
    xgb_settings = [
        "full",
        "without_dose",
        "without_pathway",
        "without_tissue",
        "without_ppi",
        "without_convergence",
        "without_complementarity",
        "without_union_coverage",
    ]
    for setting in xgb_settings:
        feature_idx = feature_groups[setting]
        tag = f"XGBoost__{setting}"
        print(f"[feature_ablation] {tag}", flush=True)
        results = _resumable_tabular_cv(tag, "XGBoost", X, y, ds["fold_splits"], xgb_params, neg_ratio, feature_idx)
        summary = summarize_tabular(results)
        rows.append(
            {
                "Model": "XGBoost",
                "Setting": setting,
                "AUROC": summary["auroc_mean"],
                "AUPRC": summary["auprc_mean"],
                "AUROC_std": summary["auroc_std"],
                "AUPRC_std": summary["auprc_std"],
                "n_folds": len(results),
            }
        )
        _write_ablation_summary(summary_path, rows)

    for setting, feature_ablation in FORMAL_FEATURE_ABLATIONS:
        tag = f"{PRIMARY_MODEL_NAME}__{setting}"
        print(f"[feature_ablation] {tag}", flush=True)
        sample_map = build_sample_collections(
            df,
            lookups,
            hp,
            ap,
            pf,
            [PRIMARY_MODEL_NAME],
            feature_ablation=feature_ablation,
        )
        cfg = ModelConfig(**champion_config.__dict__)
        cfg.neg_ratio = neg_ratio
        results = _resumable_neural_cv(tag, PRIMARY_MODEL_NAME, sample_map[PRIMARY_MODEL_NAME], y, ds["fold_splits"], cfg)
        summary = summarize_neural(results)
        rows.append(
            {
                "Model": PRIMARY_MODEL_NAME,
                "Setting": setting,
                "AUROC": summary["auroc_mean"],
                "AUPRC": summary["auprc_mean"],
                "AUROC_std": summary["auroc_std"],
                "AUPRC_std": summary["auprc_std"],
                "n_folds": len(results),
            }
        )
        _write_ablation_summary(summary_path, rows)

    _write_ablation_summary(summary_path, rows)


def run_formal_alliance_ablation(ds, df, y, hp, ap, pf, lookups, champion_config, neg_ratio: int = 10):
    """Leave-one-alliance-out ablation on DoseAwareIAM.

    Five settings (full + 4 leave-one-alliance-out) are trained from scratch on
    the same 10 fold pair-stratified CV. Each setting writes per-fold pickles
    into the active stage's fold_results/ as ``DoseAwareIAM__<setting>_fold{k}.pkl``
    and a running summary into ``tables/alliance_ablation.csv``.
    """

    print("\n== Formal Alliance-level (Leave-one-out) Ablations ==")
    rows = []
    summary_path = TABLES_DIR / "alliance_ablation.csv"

    for setting, ablation_tags in FORMAL_ALLIANCE_ABLATIONS:
        tag_label = f"{PRIMARY_MODEL_NAME}__{setting}"
        print(f"[alliance_ablation] {tag_label}", flush=True)
        sample_map = build_sample_collections(
            df,
            lookups,
            hp,
            ap,
            pf,
            [PRIMARY_MODEL_NAME],
            feature_ablation=ablation_tags or None,
        )
        cfg = ModelConfig(**champion_config.__dict__)
        cfg.neg_ratio = neg_ratio
        results = _resumable_neural_cv(
            tag_label, PRIMARY_MODEL_NAME, sample_map[PRIMARY_MODEL_NAME], y, ds["fold_splits"], cfg
        )
        summary = summarize_neural(results)
        rows.append(
            {
                "Model": PRIMARY_MODEL_NAME,
                "Setting": setting,
                "AUROC": summary["auroc_mean"],
                "AUPRC": summary["auprc_mean"],
                "AUROC_std": summary["auroc_std"],
                "AUPRC_std": summary["auprc_std"],
                "n_folds": len(results),
            }
        )
        _write_ablation_summary(summary_path, rows)

    _write_ablation_summary(summary_path, rows)


def run_formal_structure_ablation(ds, df, y, hp, ap, pf, lookups, champion_config, neg_ratio: int = 10):
    print("\n== Formal Structure-level Ablations ==")
    structure_models = DOSEAWARE_STRUCTURE_MODELS
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, structure_models)
    rows = []
    summary_path = TABLES_DIR / "structure_ablation.csv"
    fold_results_by_model: dict[str, list[dict]] = {}
    for model_name in structure_models:
        tag = model_name
        print(f"[structure_ablation] {tag}", flush=True)
        if model_name == "InteractionAwareSetModel":
            cfg = ModelConfig(hidden=32, dropout=0.3, lr=1e-3, epochs=100, patience=10, batch_size=32, neg_ratio=neg_ratio, eval_every=2)
        else:
            cfg = ModelConfig(**champion_config.__dict__)
            cfg.neg_ratio = neg_ratio
        results = _resumable_neural_cv(tag, model_name, sample_map[model_name], y, ds["fold_splits"], cfg)
        fold_results_by_model[model_name] = results
        summary = summarize_neural(results)
        rows.append(
            {
                "Model": model_name,
                "AUROC": summary["auroc_mean"],
                "AUPRC": summary["auprc_mean"],
                "AUROC_std": summary["auroc_std"],
                "AUPRC_std": summary["auprc_std"],
                "n_folds": len(results),
            }
        )
        _write_ablation_summary(summary_path, rows)
    _augment_structure_ablation_table(rows, fold_results_by_model, summary_path)


def _augment_structure_ablation_table(rows: list[dict], fold_results_by_model: dict[str, list[dict]], summary_path):
    primary = PRIMARY_MODEL_NAME
    primary_summary = next((r for r in rows if r["Model"] == primary), None)
    if primary_summary is None or primary not in fold_results_by_model:
        _write_ablation_summary(summary_path, rows)
        return
    primary_auroc = float(primary_summary["AUROC"])
    primary_auprc = float(primary_summary["AUPRC"])
    primary_auroc_per_fold = np.asarray([float(r["auroc"]) for r in fold_results_by_model[primary]], dtype=float)
    primary_auprc_per_fold = np.asarray([float(r["auprc"]) for r in fold_results_by_model[primary]], dtype=float)
    pvals_auroc: dict[str, float] = {}
    pvals_auprc: dict[str, float] = {}
    delta_auroc: dict[str, float] = {}
    delta_auprc: dict[str, float] = {}
    for row in rows:
        model = row["Model"]
        delta_auroc[model] = float(row["AUROC"]) - primary_auroc
        delta_auprc[model] = float(row["AUPRC"]) - primary_auprc
        if model == primary or model not in fold_results_by_model:
            continue
        sibling_auroc = np.asarray([float(r["auroc"]) for r in fold_results_by_model[model]], dtype=float)
        sibling_auprc = np.asarray([float(r["auprc"]) for r in fold_results_by_model[model]], dtype=float)
        if sibling_auroc.size != primary_auroc_per_fold.size or sibling_auroc.size < 2:
            pvals_auroc[model] = float("nan")
            pvals_auprc[model] = float("nan")
            continue
        from scipy import stats

        try:
            _, p_auroc = stats.ttest_rel(primary_auroc_per_fold, sibling_auroc)
        except Exception:
            p_auroc = float("nan")
        try:
            _, p_auprc = stats.ttest_rel(primary_auprc_per_fold, sibling_auprc)
        except Exception:
            p_auprc = float("nan")
        pvals_auroc[model] = float(p_auroc)
        pvals_auprc[model] = float(p_auprc)
    holm_auroc = holm_bonferroni({k: v for k, v in pvals_auroc.items() if not np.isnan(v)})
    holm_auprc = holm_bonferroni({k: v for k, v in pvals_auprc.items() if not np.isnan(v)})
    for row in rows:
        model = row["Model"]
        row["delta_AUROC_vs_primary"] = delta_auroc.get(model)
        row["delta_AUPRC_vs_primary"] = delta_auprc.get(model)
        row["p_AUROC_vs_primary"] = pvals_auroc.get(model)
        row["p_AUPRC_vs_primary"] = pvals_auprc.get(model)
        row["pHolm_AUROC_vs_primary"] = holm_auroc.get(model) if model != primary else None
        row["pHolm_AUPRC_vs_primary"] = holm_auprc.get(model) if model != primary else None
    _write_ablation_summary(summary_path, rows)


def graph_candidate_configs():
    return [
        GraphConfig(hidden=32, layers=2, heads=2, lr=1e-3, epochs=60, patience=12, eval_every=2, batch_size=256),
        GraphConfig(hidden=64, layers=2, heads=2, lr=1e-3, epochs=80, patience=16, eval_every=2, batch_size=256),
    ]


def search_graph_config(model_name, ds, split):
    best = graph_candidate_configs()[0]
    best_score = -np.inf
    sel_key = f"val_{VAL_SELECTION_METRIC}"
    for config in graph_candidate_configs():
        result = run_graph_cv(model_name, ds, [split], config, save_result=False)[0]
        if result[sel_key] > best_score:
            best_score = result[sel_key]
            best = config
    return best


def run_bakeoff(ds, X, labels, sample_map, fold_subset):
    print("\n== Bake-off ==")
    rows = []
    model_to_results = {}
    best_configs = {}

    fast_xgb_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
    }
    fast_neural_configs = {
        "AttentionPool_NoMP": ModelConfig(hidden=32, dropout=0.2, lr=5e-3, epochs=40, patience=10, eval_every=2),
        "HerbInteractionGraph": ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=40, patience=10, eval_every=2),
        "InteractionAwareSetModel": ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=40, patience=10, eval_every=2),
    }

    for model_name in ["XGBoost"]:
        params = fast_xgb_params
        best_configs[model_name] = params.copy()
        results = run_tabular_cv(model_name, X, labels, ds["fold_splits"], params, neg_ratio=1, fold_subset=fold_subset)
        model_to_results[model_name] = results
        summary = summarize_tabular(results)
        rows.append(
            {
                "Model": model_name,
                "Stage": "bakeoff",
                "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
                "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
                "AUROC_mean": summary["auroc_mean"],
                "AUPRC_mean": summary["auprc_mean"],
            }
        )

    for model_name in NEURAL_BAKEOFF_MODELS:
        config = fast_neural_configs[model_name]
        best_configs[model_name] = config.__dict__.copy()
        results = run_neural_cv(model_name, sample_map[model_name], labels, ds["fold_splits"], config, fold_subset=fold_subset)
        model_to_results[model_name] = results
        summary = summarize_neural(results)
        rows.append(
            {
                "Model": model_name,
                "Stage": "bakeoff",
                "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
                "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
                "AUROC_mean": summary["auroc_mean"],
                "AUPRC_mean": summary["auprc_mean"],
            }
        )

    sel_mean = f"{VAL_SELECTION_METRIC.upper()}_mean"
    tie_mean = "AUPRC_mean" if VAL_SELECTION_METRIC == "auroc" else "AUROC_mean"
    result_df = pd.DataFrame(rows).sort_values([sel_mean, tie_mean], ascending=False).reset_index(drop=True)
    top_models = result_df["Model"].tolist()[:2]
    champion = top_models[0]
    runner_up = top_models[1] if len(top_models) > 1 else None
    margin = float(result_df.iloc[0][sel_mean] - result_df.iloc[1][sel_mean]) if len(result_df) > 1 else np.inf
    if champion == "HerbInteractionGraph" and margin < 0.01 and "InteractionAwareSetModel" in result_df["Model"].values:
        champion = "InteractionAwareSetModel"

    artifact = {
        "results": result_df,
        "model_to_results": model_to_results,
        "best_configs": best_configs,
        "champion": champion,
        "runner_up": runner_up,
        "selection_metric": VAL_SELECTION_METRIC,
        "margin_selection": margin,
        "margin_auprc": float(result_df.iloc[0]["AUPRC_mean"] - result_df.iloc[1]["AUPRC_mean"]) if len(result_df) > 1 else np.inf,
        "fold_subset": fold_subset,
    }
    save_pickle(artifact, CHAMPION_PATH)
    result_df.to_csv(TABLES_DIR / "bakeoff_results.csv", index=False)
    print(result_df[["Model", "AUROC", "AUPRC"]].to_string(index=False))
    print(f"\nChampion: {champion} (runner-up: {runner_up}, delta {VAL_SELECTION_METRIC.upper()}={margin:.4f})")
    return artifact


def run_main_benchmark(
    ds,
    X,
    labels,
    sample_map,
    champion_artifact,
    neg_ratio: int = 1,
    tabular_models: list[str] | None = None,
    neural_models: list[str] | None = None,
    graph_models: list[str] | None = None,
    reference_model: str | None = None,
    sort_metric: str = "auprc",
    fold_subset: list[int] | None = None,
):
    print("\n== Main Benchmark ==")
    best_configs = resolve_best_configs(champion_artifact)
    champion = champion_artifact["champion"]
    reference_model = reference_model or champion
    tabular_models = TABULAR_MODELS if tabular_models is None else tabular_models
    neural_models = [champion] if neural_models is None else neural_models
    graph_models = GRAPH_MAIN_MODELS if graph_models is None else graph_models
    rows = []
    model_to_results = {}

    for model_name in tabular_models:
        params = best_configs.get(model_name) or search_params(model_name, X, labels, ds["fold_splits"][0], neg_ratio=neg_ratio)
        results = run_tabular_cv(model_name, X, labels, ds["fold_splits"], params, neg_ratio=neg_ratio, fold_subset=fold_subset)
        model_to_results[model_name] = results
        summary = summarize_tabular(results)
        rows.append(
            {
                "Model": model_name,
                "Group": "A",
                "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
                "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
                "AUROC_mean": summary["auroc_mean"],
                "AUPRC_mean": summary["auprc_mean"],
            }
        )

    for model_name in neural_models:
        cfg = resolve_neural_config(model_name, sample_map, labels, ds["fold_splits"], best_configs, neg_ratio=neg_ratio)
        results = run_neural_cv(model_name, sample_map[model_name], labels, ds["fold_splits"], cfg, fold_subset=fold_subset)
        model_to_results[model_name] = results
        summary = summarize_neural(results)
        rows.append(
            {
                "Model": model_name,
                "Group": "D",
                "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
                "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
                "AUROC_mean": summary["auroc_mean"],
                "AUPRC_mean": summary["auprc_mean"],
            }
        )

    graph_configs = {}
    graph_splits = ds["fold_splits"]
    if fold_subset is not None:
        graph_splits = [split for split in ds["fold_splits"] if int(split.get("fold", split.get("seed", 0))) in fold_subset]
    for model_name in graph_models:
        cfg = search_graph_config(model_name, ds, ds["fold_splits"][0])
        graph_configs[model_name] = cfg
        results = run_graph_cv(model_name, ds, graph_splits, cfg)
        model_to_results[model_name] = results
        summary = summarize_graph(results)
        group = "C" if model_name == "R-GCN (w/ Formula)" else "B"
        rows.append(
            {
                "Model": model_name,
                "Group": group,
                "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
                "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
                "AUROC_mean": summary["auroc_mean"],
                "AUPRC_mean": summary["auprc_mean"],
            }
        )

    sort_metric = sort_metric.lower()
    sort_cols = ["AUROC_mean", "AUPRC_mean"] if sort_metric == "auroc" else ["AUPRC_mean", "AUROC_mean"]
    benchmark_df = pd.DataFrame(rows).sort_values(sort_cols, ascending=False).reset_index(drop=True)

    pvals = {}
    if reference_model in model_to_results:
        reference_scores = [r["auprc"] for r in model_to_results[reference_model]]
        for model_name, results in model_to_results.items():
            if model_name == reference_model:
                continue
            lhs = np.asarray(reference_scores, dtype=float)
            rhs = np.asarray([r["auprc"] for r in results], dtype=float)
            from scipy import stats

            _, pvalue = stats.ttest_rel(lhs, rhs)
            pvals[model_name] = float(pvalue)
    pvals_holm = holm_bonferroni(pvals)
    benchmark_df["pHolm_AUPRC_vs_reference"] = benchmark_df["Model"].map(pvals_holm)
    benchmark_df.to_csv(TABLES_DIR / "main_benchmark.csv", index=False)

    pmat_auc = paired_ttest_matrix(model_to_results, "auroc")
    pmat_auprc = paired_ttest_matrix(model_to_results, "auprc")
    pmat_auc.to_csv(TABLES_DIR / "significance_matrix_auroc.csv")
    pmat_auprc.to_csv(TABLES_DIR / "significance_matrix_auprc.csv")

    pooled_rows = []
    for model_name, results in model_to_results.items():
        y_true, y_prob = compute_pooled_predictions(results)
        pooled_rows.append({"Model": model_name, "n": len(y_true), "pooled_auroc": compute_metrics(y_true, y_prob)["auroc"]})
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_tests = []
    if reference_model in model_to_results:
        reference_true, reference_prob = compute_pooled_predictions(model_to_results[reference_model])
        for model_name, results in model_to_results.items():
            if model_name == reference_model:
                continue
            y_true, y_prob = compute_pooled_predictions(results)
            p_delong = delong_test(reference_true, reference_prob, y_prob)
            pooled_tests.append({"reference_model": reference_model, "baseline": model_name, "delong_p_auroc": p_delong})
    pd.DataFrame(pooled_tests).to_csv(SUPP_DIR / "pooled_significance.csv", index=False)

    pooled_df.to_csv(EVAL_SUMMARY_PATH, index=False)
    if graph_configs:
        save_pickle({"graph_configs": {k: v.__dict__ for k, v in graph_configs.items()}}, SUPP_DIR / "graph_hyperparameters.pkl")
    print(benchmark_df[["Model", "Group", "AUROC", "AUPRC"]].to_string(index=False))
    return reference_model, best_configs, model_to_results


def run_feature_ablation(ds, X, y, champion, champion_config, sample_map, neg_ratio: int = 1):
    print("\n== Feature and Architecture Ablations ==")
    feature_groups = build_feature_groups(ds["feature_cols"])
    rows = []

    xgb_params = search_params("XGBoost", X, y, ds["fold_splits"][0], neg_ratio=neg_ratio)
    for name, feature_idx in feature_groups.items():
        results = run_tabular_cv("XGBoost", X, y, ds["fold_splits"], xgb_params, neg_ratio=neg_ratio, feature_idx=feature_idx)
        summary = summarize_tabular(results)
        rows.append({"Model": "XGBoost", "Setting": name, "AUROC": summary["auroc_mean"], "AUPRC": summary["auprc_mean"]})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "feature_ablation.csv", index=False)

    arch_rows = []
    if champion == "DoseAwareIAM":
        model_list = ["DoseAwareIAM", "InteractionAwareSetModel"]
    else:
        winner_family = NEURAL_ABLATION_MODELS if champion in {"HerbInteractionGraph", "InteractionAwareSetModel", "AttentionPool_NoMP"} else []
        model_list = [champion] + [m for m in winner_family if m != champion]
    for model_name in model_list:
        try:
            cfg = ModelConfig(**champion_config.__dict__)
            cfg.neg_ratio = neg_ratio
            results = run_neural_cv(model_name, sample_map[model_name], y, ds["fold_splits"], cfg)
            summary = summarize_neural(results)
            arch_rows.append({"Model": model_name, "AUROC": summary["auroc_mean"], "AUPRC": summary["auprc_mean"]})
        except Exception as exc:
            arch_rows.append({"Model": model_name, "AUROC": np.nan, "AUPRC": np.nan, "error": str(exc)})
    pd.DataFrame(arch_rows).to_csv(TABLES_DIR / "architecture_ablation.csv", index=False)


COLD_START_TABULAR_MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP", "XGBoost"]
COLD_START_NEURAL_MODELS = [PRIMARY_MODEL_NAME]
COLD_START_GRAPH_MODELS = ["R-GCN", "HGT"]
COLD_START_SPLIT_TYPES = ["Formula", "ADR"]


def _cold_fold_path(split_type: str, seed: int, model_name: str):
    from experiment_utils import FOLD_RESULTS_DIR, sanitize_name

    return FOLD_RESULTS_DIR / f"cold_{split_type.lower()}_{int(seed)}_{sanitize_name(model_name)}.pkl"


def _cold_select_splits(splits: list[dict], seed_subset: list[int] | None) -> list[dict]:
    if seed_subset is None:
        return splits
    return [s for s in splits if int(s["seed"]) in seed_subset]


def _resumable_cold_tabular(
    model_name: str,
    X,
    y,
    splits: list[dict],
    split_type: str,
    params: dict,
    neg_ratio: int,
):
    from experiment_utils import load_pickle, save_pickle

    out = []
    for split in splits:
        seed = int(split["seed"])
        path = _cold_fold_path(split_type, seed, model_name)
        if path.exists():
            out.append(load_pickle(path))
            continue
        result = fit_predict_split(model_name, X, y, split, params, neg_ratio=neg_ratio, seed=42)
        result["cold_start_split"] = split_type
        result["cold_start_seed"] = seed
        save_pickle(result, path)
        out.append(result)
    return out


def _resumable_cold_neural(
    model_name: str,
    samples,
    y,
    splits: list[dict],
    split_type: str,
    cfg: ModelConfig,
    neg_ratio: int,
):
    from experiment_utils import load_pickle, save_pickle

    out = []
    cfg = ModelConfig(**cfg.__dict__)
    cfg.neg_ratio = int(neg_ratio)
    for split in splits:
        seed = int(split["seed"])
        path = _cold_fold_path(split_type, seed, model_name)
        if path.exists():
            out.append(load_pickle(path))
            continue
        result = train_neural_split(model_name, samples, y, split, cfg, save_model=False)
        result["cold_start_split"] = split_type
        result["cold_start_seed"] = seed
        save_pickle(result, path)
        out.append(result)
    return out


def _resumable_cold_graph(
    model_name: str,
    ds,
    splits: list[dict],
    split_type: str,
    cfg: GraphConfig,
    neg_ratio: int,
):
    from experiment_utils import load_pickle, save_pickle
    from graph_baselines import train_one_split as train_graph_split

    out = []
    cfg = GraphConfig(**cfg.__dict__)
    cfg.neg_ratio = int(neg_ratio)
    for split in splits:
        seed = int(split["seed"])
        path = _cold_fold_path(split_type, seed, model_name)
        if path.exists():
            out.append(load_pickle(path))
            continue
        result = train_graph_split(model_name, ds, split, cfg, save_result=False)
        result["cold_start_split"] = split_type
        result["cold_start_seed"] = seed
        save_pickle(result, path)
        out.append(result)
    return out


def _summarize_cold_results(results: list[dict]) -> dict:
    if not results:
        return {f"{k}_mean": float("nan") for k in ["auroc", "auprc", "precision", "recall", "f1", "mcc"]}
    summary = {}
    for k in ["auroc", "auprc", "precision", "recall", "f1", "mcc"]:
        vals = np.asarray([float(r[k]) for r in results], dtype=float)
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=0))
    return summary


def run_cold_start(
    ds,
    X,
    y,
    sample_map,
    best_configs,
    neg_ratio: int = 10,
    tabular_models: list[str] | None = None,
    neural_models: list[str] | None = None,
    graph_models: list[str] | None = None,
    seed_subset: list[int] | None = None,
    split_types: list[str] | None = None,
):
    print("\n== Cold-start ==")
    tabular_models = COLD_START_TABULAR_MODELS if tabular_models is None else tabular_models
    neural_models = COLD_START_NEURAL_MODELS if neural_models is None else neural_models
    graph_models = COLD_START_GRAPH_MODELS if graph_models is None else graph_models
    split_types = COLD_START_SPLIT_TYPES if split_types is None else split_types
    splits_lookup = {
        "Formula": ds.get("formula_cs_splits", []),
        "ADR": ds.get("adr_cs_splits", []),
    }
    tabular_params = {
        model_name: best_configs.get(model_name) or search_params(model_name, X, y, ds["fold_splits"][0], neg_ratio=neg_ratio)
        for model_name in tabular_models
    }
    rows: list[dict] = []
    for split_type in split_types:
        active_splits = _cold_select_splits(splits_lookup.get(split_type, []), seed_subset)
        for model_name in tabular_models:
            print(f"[cold_start] {split_type} CS, tabular {model_name}", flush=True)
            results = _resumable_cold_tabular(model_name, X, y, active_splits, split_type, tabular_params[model_name], neg_ratio)
            for r in results:
                rows.append({
                    "split_type": split_type,
                    "seed": int(r["cold_start_seed"]),
                    "model": model_name,
                    "type": "tabular",
                    "auroc": float(r["auroc"]),
                    "auprc": float(r["auprc"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                    "mcc": float(r["mcc"]),
                    "n_test": int(r.get("n_test", len(r.get("y_true", [])))),
                })
        for model_name in neural_models:
            if model_name not in sample_map:
                print(f"  WARN: neural model {model_name} missing in sample_map; skipping cold-start.")
                continue
            print(f"[cold_start] {split_type} CS, neural {model_name}", flush=True)
            cfg = resolve_neural_config(model_name, sample_map, y, ds["fold_splits"], best_configs, neg_ratio=neg_ratio)
            results = _resumable_cold_neural(model_name, sample_map[model_name], y, active_splits, split_type, cfg, neg_ratio)
            for r in results:
                rows.append({
                    "split_type": split_type,
                    "seed": int(r["cold_start_seed"]),
                    "model": model_name,
                    "type": "neural",
                    "auroc": float(r["auroc"]),
                    "auprc": float(r["auprc"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                    "mcc": float(r["mcc"]),
                    "n_test": int(r.get("n_test", len(r.get("y_true", [])))),
                })
        if graph_models:
            graph_cfg = graph_candidate_configs()[0]
            for model_name in graph_models:
                print(f"[cold_start] {split_type} CS, graph {model_name}", flush=True)
                results = _resumable_cold_graph(model_name, ds, active_splits, split_type, graph_cfg, neg_ratio)
                for r in results:
                    rows.append({
                        "split_type": split_type,
                        "seed": int(r["cold_start_seed"]),
                        "model": model_name,
                        "type": "graph",
                        "auroc": float(r["auroc"]),
                        "auprc": float(r["auprc"]),
                        "precision": float(r["precision"]),
                        "recall": float(r["recall"]),
                        "f1": float(r["f1"]),
                        "mcc": float(r["mcc"]),
                        "n_test": int(r.get("n_test", len(r.get("y_true", [])))),
                    })

    progress_path = SUPP_DIR / "cold_start_progress.csv"
    pd.DataFrame(rows).sort_values(["split_type", "model", "seed"]).to_csv(progress_path, index=False)

    agg_rows: list[dict] = []
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_pair.setdefault((row["split_type"], row["model"]), []).append(row)
    for (split_type, model_name), entries in by_pair.items():
        summary = _summarize_cold_results(entries)
        agg_rows.append({
            "Model": model_name,
            "split_type": split_type,
            "n_seeds": len(entries),
            "AUROC_mean": summary["auroc_mean"],
            "AUROC_std": summary["auroc_std"],
            "AUPRC_mean": summary["auprc_mean"],
            "AUPRC_std": summary["auprc_std"],
            "F1_mean": summary["f1_mean"],
            "MCC_mean": summary["mcc_mean"],
        })
    out_df = pd.DataFrame(agg_rows).sort_values(["split_type", "Model"]).reset_index(drop=True)
    out_df.to_csv(TABLES_DIR / "cold_start.csv", index=False)
    print(f"\nWrote {TABLES_DIR / 'cold_start.csv'} ({len(out_df)} rows) and {progress_path} ({len(rows)} per-cell rows)")
    return out_df


NEG_RATIO_GRID = [1, 3, 5, 10]
NEG_SENSITIVITY_TABULAR = ["XGBoost", "GradientBoosting"]
NEG_SENSITIVITY_NEURAL = [PRIMARY_MODEL_NAME]
NEG_SENSITIVITY_GRAPH = ["R-GCN"]
NEG_SENSITIVITY_PRIMARY = PRIMARY_MODEL_NAME


def _neg_sensitivity_fold_path(model_name: str, neg_ratio: int, fold_id: int):
    from experiment_utils import FOLD_RESULTS_DIR, sanitize_name

    return FOLD_RESULTS_DIR / f"{sanitize_name(model_name)}_neg{int(neg_ratio)}_fold{int(fold_id)}.pkl"


def _resumable_neg_tabular_cv(model_name: str, X, y, fold_splits, neg_ratio: int, params: dict):
    from experiment_utils import load_pickle, save_pickle

    results = []
    for split in fold_splits:
        fold_id = int(split.get("fold", split.get("seed", 0)))
        path = _neg_sensitivity_fold_path(model_name, neg_ratio, fold_id)
        if path.exists():
            results.append(load_pickle(path))
            continue
        result = fit_predict_split(model_name, X, y, split, params, neg_ratio=neg_ratio, seed=42)
        result["neg_ratio"] = int(neg_ratio)
        result["sensitivity_tag"] = f"{model_name}_neg{neg_ratio}"
        save_pickle(result, path)
        results.append(result)
    return sorted(results, key=lambda r: int(r.get("fold", 0)))


def _resumable_neg_neural_cv(model_name: str, samples, y, fold_splits, neg_ratio: int, base_cfg: ModelConfig):
    from experiment_utils import load_pickle, save_pickle

    results = []
    cfg = ModelConfig(**base_cfg.__dict__)
    cfg.neg_ratio = int(neg_ratio)
    for split in fold_splits:
        fold_id = int(split.get("fold", split.get("seed", 0)))
        path = _neg_sensitivity_fold_path(model_name, neg_ratio, fold_id)
        if path.exists():
            results.append(load_pickle(path))
            continue
        result = train_neural_split(model_name, samples, y, split, cfg, save_model=False)
        result["neg_ratio"] = int(neg_ratio)
        result["sensitivity_tag"] = f"{model_name}_neg{neg_ratio}"
        save_pickle(result, path)
        results.append(result)
    return sorted(results, key=lambda r: int(r.get("fold", 0)))


def _resumable_neg_graph_cv(model_name: str, ds, fold_splits, neg_ratio: int, cfg: GraphConfig):
    from experiment_utils import load_pickle, save_pickle
    from graph_baselines import train_one_split as train_graph_split

    results = []
    cfg = GraphConfig(**cfg.__dict__)
    cfg.neg_ratio = int(neg_ratio)
    for split in fold_splits:
        fold_id = int(split.get("fold", split.get("seed", 0)))
        path = _neg_sensitivity_fold_path(model_name, neg_ratio, fold_id)
        if path.exists():
            results.append(load_pickle(path))
            continue
        result = train_graph_split(model_name, ds, split, cfg, save_result=False)
        result["neg_ratio"] = int(neg_ratio)
        result["sensitivity_tag"] = f"{model_name}_neg{neg_ratio}"
        save_pickle(result, path)
        results.append(result)
    return sorted(results, key=lambda r: int(r.get("fold", 0)))


def run_neg_sampling_sensitivity(
    ds,
    X,
    y,
    champion,
    sample_map,
    champion_config: ModelConfig,
    tabular_models: list[str] | None = None,
    neural_models: list[str] | None = None,
    graph_models: list[str] | None = None,
    ratios: list[int] | None = None,
    fold_subset: list[int] | None = None,
):
    print("\n== Negative Sampling Sensitivity ==")
    tabular_models = NEG_SENSITIVITY_TABULAR if tabular_models is None else tabular_models
    neural_models = NEG_SENSITIVITY_NEURAL if neural_models is None else neural_models
    graph_models = NEG_SENSITIVITY_GRAPH if graph_models is None else graph_models
    ratios = NEG_RATIO_GRID if ratios is None else ratios
    fold_splits = ds["fold_splits"]
    if fold_subset is not None:
        fold_splits = [s for s in fold_splits if int(s.get("fold", s.get("seed", 0))) in fold_subset]

    aggregated: dict[tuple[str, int], list[dict]] = {}
    rows = []

    tabular_param_cache: dict[str, dict] = {}
    for model_name in tabular_models:
        if model_name not in tabular_param_cache:
            tabular_param_cache[model_name] = search_params(model_name, X, y, ds["fold_splits"][0], neg_ratio=10)
        for ratio in ratios:
            print(f"[neg_sens] tabular {model_name} ratio=1:{ratio}", flush=True)
            results = _resumable_neg_tabular_cv(model_name, X, y, fold_splits, ratio, tabular_param_cache[model_name])
            aggregated[(model_name, ratio)] = results
            summary = summarize_tabular(results)
            rows.append({
                "model": model_name,
                "type": "tabular",
                "neg_ratio": int(ratio),
                "n_folds": len(results),
                "auroc_mean": summary["auroc_mean"],
                "auprc_mean": summary["auprc_mean"],
                "auroc_std": summary["auroc_std"],
                "auprc_std": summary["auprc_std"],
            })

    for model_name in neural_models:
        if model_name not in sample_map:
            print(f"  WARN: neural model {model_name} missing from sample_map; skipping.")
            continue
        for ratio in ratios:
            print(f"[neg_sens] neural {model_name} ratio=1:{ratio}", flush=True)
            results = _resumable_neg_neural_cv(model_name, sample_map[model_name], y, fold_splits, ratio, champion_config)
            aggregated[(model_name, ratio)] = results
            summary = summarize_neural(results)
            rows.append({
                "model": model_name,
                "type": "neural",
                "neg_ratio": int(ratio),
                "n_folds": len(results),
                "auroc_mean": summary["auroc_mean"],
                "auprc_mean": summary["auprc_mean"],
                "auroc_std": summary["auroc_std"],
                "auprc_std": summary["auprc_std"],
            })

    if graph_models:
        graph_cfg = graph_candidate_configs()[0]
        for model_name in graph_models:
            for ratio in ratios:
                print(f"[neg_sens] graph {model_name} ratio=1:{ratio}", flush=True)
                results = _resumable_neg_graph_cv(model_name, ds, fold_splits, ratio, graph_cfg)
                aggregated[(model_name, ratio)] = results
                summary = summarize_graph(results)
                rows.append({
                    "model": model_name,
                    "type": "graph",
                    "neg_ratio": int(ratio),
                    "n_folds": len(results),
                    "auroc_mean": summary["auroc_mean"],
                    "auprc_mean": summary["auprc_mean"],
                    "auroc_std": summary["auroc_std"],
                    "auprc_std": summary["auprc_std"],
                })

    primary = NEG_SENSITIVITY_PRIMARY
    pivot_ratio = 10
    primary_pivot = aggregated.get((primary, pivot_ratio))
    if primary_pivot is None:
        for row in rows:
            row["p_AUROC_vs_primary_neg10"] = float("nan")
            row["p_AUPRC_vs_primary_neg10"] = float("nan")
    else:
        from scipy import stats

        primary_auroc_per_fold = np.asarray([float(r["auroc"]) for r in primary_pivot], dtype=float)
        primary_auprc_per_fold = np.asarray([float(r["auprc"]) for r in primary_pivot], dtype=float)
        for row in rows:
            sibling = aggregated.get((row["model"], int(row["neg_ratio"])))
            if sibling is None or len(sibling) != len(primary_pivot):
                row["p_AUROC_vs_primary_neg10"] = float("nan")
                row["p_AUPRC_vs_primary_neg10"] = float("nan")
                continue
            sib_auroc = np.asarray([float(r["auroc"]) for r in sibling], dtype=float)
            sib_auprc = np.asarray([float(r["auprc"]) for r in sibling], dtype=float)
            try:
                _, p1 = stats.ttest_rel(primary_auroc_per_fold, sib_auroc)
            except Exception:
                p1 = float("nan")
            try:
                _, p2 = stats.ttest_rel(primary_auprc_per_fold, sib_auprc)
            except Exception:
                p2 = float("nan")
            row["p_AUROC_vs_primary_neg10"] = float(p1)
            row["p_AUPRC_vs_primary_neg10"] = float(p2)

    out_df = pd.DataFrame(rows).sort_values(["model", "neg_ratio"]).reset_index(drop=True)
    out_path = SUPP_DIR / "neg_sampling_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df)} rows)")
    return out_df


def run_compute_profile(model_results):
    rows = []
    for model_name, results in model_results.items():
        train_times = np.asarray([r["train_time_sec"] for r in results], dtype=float)
        infer_times = np.asarray([r["inference_time_sec"] for r in results], dtype=float)
        mems = np.asarray([r["peak_memory_mb"] for r in results], dtype=float)
        rows.append(
            {
                "model": model_name,
                "training_time_per_fold_sec": float(np.nanmean(train_times)),
                "inference_time_sec": float(np.nanmean(infer_times)),
                "peak_memory_mb": float(np.nanmean(mems)),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "computational_profile.csv", index=False)


def run_calibration(model_name, model_results):
    y_true, y_prob = compute_pooled_predictions(model_results[model_name])
    calib = calibration_table(y_true, y_prob)
    calib["model"] = model_name
    calib.to_csv(SUPP_DIR / "calibration.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["bakeoff", "benchmark", "quick", "all"])
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--include-models", default="", help="Comma-separated benchmark model allowlist for quick mode.")
    parser.add_argument("--reference-model", default=PRIMARY_MODEL_NAME, help="Reference neural model for quick mode reporting.")
    parser.add_argument("--fold-subset", default="", help="Comma-separated fold ids to evaluate, useful for quick screening.")
    parser.add_argument("--cold-start-seeds", default="", help="Comma-separated cold-start seeds to evaluate, useful for quick screening.")
    parser.add_argument("--fast-configs", action="store_true", help="Use reduced neural configs for exploratory screening only.")
    args = parser.parse_args()

    ds, df, feature_cols, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    fold_subset = [0, 1, 2]
    quick_models = parse_model_allowlist(args.include_models, QUICK_DEFAULT_MODELS)
    if args.reference_model and args.reference_model not in quick_models:
        quick_models = [args.reference_model] + quick_models
    quick_tabular, quick_neural, quick_graph = split_selected_models(quick_models)
    fold_subset_arg = parse_int_list(args.fold_subset)
    cold_start_seed_subset = parse_int_list(args.cold_start_seeds)

    if args.mode == "quick":
        sample_map = build_sample_collections(df, lookups, hp, ap, pf, quick_neural)
        best_cfgs = quick_best_configs() if args.fast_configs else default_best_configs()
        quick_artifact = {"best_configs": best_cfgs, "champion": args.reference_model}
        reference_model, best_configs, model_results = run_main_benchmark(
            ds,
            X,
            labels,
            sample_map,
            quick_artifact,
            neg_ratio=args.neg_ratio,
            tabular_models=quick_tabular,
            neural_models=quick_neural,
            graph_models=quick_graph,
            reference_model=args.reference_model,
            sort_metric="auroc",
            fold_subset=fold_subset_arg,
        )
        run_cold_start(
            ds,
            X,
            labels,
            sample_map,
            best_configs,
            neg_ratio=args.neg_ratio,
            tabular_models=quick_tabular,
            neural_models=quick_neural,
            seed_subset=cold_start_seed_subset,
        )
        print("\nPhase 4 quick run complete.")
        return

    sample_map = build_sample_collections(df, lookups, hp, ap, pf, NEURAL_BAKEOFF_MODELS + NEURAL_ABLATION_MODELS)
    if args.mode == "benchmark":
        if CHAMPION_PATH.exists():
            champion_artifact = load_pickle(CHAMPION_PATH)
            if "best_configs" not in champion_artifact:
                champion_artifact["best_configs"] = default_best_configs()
        else:
            champion_artifact = run_bakeoff(ds, X, labels, sample_map, fold_subset)
    else:
        champion_artifact = run_bakeoff(ds, X, labels, sample_map, fold_subset)
    if args.mode == "bakeoff":
        return

    champion, best_configs, model_results = run_main_benchmark(
        ds,
        X,
        labels,
        sample_map,
        champion_artifact,
        neg_ratio=args.neg_ratio,
        reference_model=champion_artifact["champion"],
        fold_subset=fold_subset_arg,
    )
    champion_config = resolve_neural_config(champion, sample_map, labels, ds["fold_splits"], best_configs, neg_ratio=args.neg_ratio)
    run_feature_ablation(ds, X, labels, champion, champion_config, sample_map, neg_ratio=args.neg_ratio)
    run_cold_start(
        ds,
        X,
        labels,
        sample_map,
        best_configs,
        neg_ratio=args.neg_ratio,
        tabular_models=["XGBoost"],
        neural_models=[champion],
        seed_subset=cold_start_seed_subset,
    )
    run_calibration(champion, model_results)
    run_neg_sampling_sensitivity(ds, X, labels, champion, sample_map, champion_config)
    run_compute_profile(model_results)
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
