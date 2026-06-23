from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experiment_utils import FULL_EXPERIMENT_DIR, RESULTS_ROOT_DIR, ensure_output_dirs
from neural_models import ModelConfig, build_sample_collections
from phase4_evaluation import (
    DOSEAWARE_STRUCTURE_MODELS,
    NEURAL_ABLATION_MODELS,
    PRIMARY_MODEL_NAME,
    default_best_configs,
    prepare_common_inputs,
    resolve_neural_config,
    run_calibration,
    run_cold_start,
    run_compute_profile,
    run_formal_alliance_ablation,
    run_formal_feature_ablation,
    run_formal_structure_ablation,
    run_main_benchmark,
    run_neg_sampling_sensitivity,
)


sys.stdout.reconfigure(line_buffering=True)
SRC_DIR = Path(__file__).resolve().parent.parent


FROZEN_DOSEAWARE_CONFIG = ModelConfig(
    hidden=32,
    dropout=0.3,
    lr=1e-3,
    epochs=100,
    patience=10,
    batch_size=32,
    neg_ratio=10,
    eval_every=2,
)


def _frozen_config(neg_ratio: int) -> ModelConfig:
    cfg = ModelConfig(**FROZEN_DOSEAWARE_CONFIG.__dict__)
    cfg.neg_ratio = neg_ratio
    return cfg


def _prepare_context(primary_only_neural: bool):
    ensure_output_dirs()
    print(f"[formal] results_root={RESULTS_ROOT_DIR}", flush=True)
    print(f"[formal] stage_dir={FULL_EXPERIMENT_DIR}", flush=True)
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    if primary_only_neural:
        sample_models = [PRIMARY_MODEL_NAME]
    else:
        sample_models = list(
            dict.fromkeys([PRIMARY_MODEL_NAME, "InteractionAwareSetModel"] + NEURAL_ABLATION_MODELS + DOSEAWARE_STRUCTURE_MODELS)
        )
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, sample_models)
    return ds, df, feature_cols, X, labels, hp, ap, pf, lookups, sample_map


def _run_main_benchmark_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, sample_map = _prepare_context(primary_only_neural=True)
    formal_artifact = {"champion": PRIMARY_MODEL_NAME, "best_configs": default_best_configs()}
    run_main_benchmark(
        ds,
        X,
        labels,
        sample_map,
        formal_artifact,
        neg_ratio=neg_ratio,
        reference_model=PRIMARY_MODEL_NAME,
        sort_metric="auroc",
    )
    print("\n[formal] main benchmark complete.", flush=True)


def _run_feature_ablation_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, _ = _prepare_context(primary_only_neural=True)
    cfg = _frozen_config(neg_ratio)
    run_formal_feature_ablation(ds, df, X, labels, hp, ap, pf, lookups, cfg, neg_ratio=neg_ratio)
    print("\n[formal] feature ablation complete.", flush=True)


def _run_structure_ablation_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, _ = _prepare_context(primary_only_neural=False)
    cfg = _frozen_config(neg_ratio)
    run_formal_structure_ablation(ds, df, labels, hp, ap, pf, lookups, cfg, neg_ratio=neg_ratio)
    print("\n[formal] structure ablation complete.", flush=True)


def _run_alliance_ablation_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, _ = _prepare_context(primary_only_neural=True)
    cfg = _frozen_config(neg_ratio)
    run_formal_alliance_ablation(ds, df, labels, hp, ap, pf, lookups, cfg, neg_ratio=neg_ratio)
    print("\n[formal] alliance ablation complete.", flush=True)


def _run_cold_start_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, sample_map = _prepare_context(primary_only_neural=True)
    best_configs = default_best_configs()
    run_cold_start(
        ds,
        X,
        labels,
        sample_map,
        best_configs,
        neg_ratio=neg_ratio,
    )
    print("\n[formal] cold-start complete.", flush=True)


def _run_calibration_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, sample_map = _prepare_context(primary_only_neural=True)
    formal_artifact = {"champion": PRIMARY_MODEL_NAME, "best_configs": default_best_configs()}
    _, _, model_results = run_main_benchmark(
        ds,
        X,
        labels,
        sample_map,
        formal_artifact,
        neg_ratio=neg_ratio,
        reference_model=PRIMARY_MODEL_NAME,
        sort_metric="auroc",
    )
    run_calibration(PRIMARY_MODEL_NAME, model_results)
    print("\n[formal] calibration complete.", flush=True)


def _run_neg_sensitivity_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, sample_map = _prepare_context(primary_only_neural=True)
    cfg = _frozen_config(neg_ratio)
    run_neg_sampling_sensitivity(
        ds,
        X,
        labels,
        PRIMARY_MODEL_NAME,
        sample_map,
        cfg,
    )
    print("\n[formal] neg-sampling sensitivity complete.", flush=True)


def _run_compute_profile_stage(neg_ratio: int):
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups, sample_map = _prepare_context(primary_only_neural=True)
    formal_artifact = {"champion": PRIMARY_MODEL_NAME, "best_configs": default_best_configs()}
    _, _, model_results = run_main_benchmark(
        ds,
        X,
        labels,
        sample_map,
        formal_artifact,
        neg_ratio=neg_ratio,
        reference_model=PRIMARY_MODEL_NAME,
        sort_metric="auroc",
    )
    run_compute_profile(model_results)
    print("\n[formal] compute profile complete.", flush=True)


def _delegate(script: str, extra_args: list[str] | None = None) -> int:
    cmd = [sys.executable, "-u", str(SRC_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[formal] delegating to: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=os.environ).returncode


def _run_supplementary_analyses_stage(neg_ratio: int):
    rc = _delegate("supplementary_analyses.py", ["all"])
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] supplementary analyses complete.", flush=True)


def _run_interpretability_stage(neg_ratio: int):
    rc = _delegate("phase5_interpretability.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] interpretability complete.", flush=True)


def _run_figures_stage(neg_ratio: int):
    rc = _delegate("make_figures.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] figures complete.", flush=True)


def _run_bootstrap_ci_stage(neg_ratio: int):
    rc = _delegate("bootstrap_pooled_ci.py", ["--n-boot", "1000"])
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] bootstrap CI complete.", flush=True)


def _run_compute_profile_postprocess_stage(neg_ratio: int):
    rc = _delegate("compute_profile_postprocess.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] compute profile postprocess complete.", flush=True)


def _run_aggregate_structure_stage(neg_ratio: int):
    rc = _delegate("aggregate_formal_structure_ablation.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] structure-ablation aggregate complete.", flush=True)


def _run_aggregate_neg_sens_stage(neg_ratio: int):
    rc = _delegate("aggregate_formal_neg_sensitivity.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] neg-sensitivity aggregate complete.", flush=True)


def _run_aggregate_cold_start_stage(neg_ratio: int):
    rc = _delegate("aggregate_formal_cold_start.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] cold-start aggregate complete.", flush=True)


def _run_aggregate_alliance_stage(neg_ratio: int):
    rc = _delegate("aggregate_formal_alliance_ablation.py")
    if rc != 0:
        raise SystemExit(rc)
    print("\n[formal] alliance ablation aggregate complete.", flush=True)


STAGES = {
    "main_benchmark": _run_main_benchmark_stage,
    "feature_ablation": _run_feature_ablation_stage,
    "structure_ablation": _run_structure_ablation_stage,
    "alliance_ablation": _run_alliance_ablation_stage,
    "cold_start": _run_cold_start_stage,
    "calibration": _run_calibration_stage,
    "neg_sampling_sensitivity": _run_neg_sensitivity_stage,
    "compute_profile": _run_compute_profile_stage,
    "compute_profile_postprocess": _run_compute_profile_postprocess_stage,
    "supplementary_analyses": _run_supplementary_analyses_stage,
    "interpretability": _run_interpretability_stage,
    "figures": _run_figures_stage,
    "bootstrap_ci": _run_bootstrap_ci_stage,
    "aggregate_structure_ablation": _run_aggregate_structure_stage,
    "aggregate_neg_sampling_sensitivity": _run_aggregate_neg_sens_stage,
    "aggregate_cold_start": _run_aggregate_cold_start_stage,
    "aggregate_alliance_ablation": _run_aggregate_alliance_stage,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=list(STAGES.keys()))
    parser.add_argument("--neg-ratio", type=int, default=10)
    args = parser.parse_args()
    print(f"[formal] stage={args.stage} | primary_model={PRIMARY_MODEL_NAME} | neg_ratio={args.neg_ratio}", flush=True)
    STAGES[args.stage](args.neg_ratio)


if __name__ == "__main__":
    main()
