from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import format_metric
from neural_models import ModelConfig, build_sample_collections, summarize_results as summarize_neural, train_one_split
from phase4_evaluation import prepare_common_inputs


sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "doseaware_sensitivity_val_auroc"
FOLD_DIR = ROOT / "fold_results"
TABLE_DIR = ROOT / "tables"
BASELINE_SOURCE = Path(__file__).resolve().parent.parent.parent / "outputs" / "accurate_neg10_val_auroc_variants" / "fold_results"
PRIMARY_MODEL = "DoseAwareIAM"


def ensure_dirs():
    for path in [ROOT, FOLD_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def config_tag(cfg: ModelConfig) -> str:
    lr_tag = (
        f"{cfg.lr:.0e}"
        .replace("e-0", "e-")
        .replace("e+0", "e+")
        .replace("e+", "e")
    )
    return f"h{cfg.hidden}_d{cfg.dropout:.1f}_lr{lr_tag}_e{cfg.epochs}_p{cfg.patience}"


def config_row(axis: str, setting: str, cfg: ModelConfig) -> dict:
    return {
        "axis": axis,
        "setting": setting,
        "config": cfg,
        "tag": config_tag(cfg),
    }


def candidate_configs() -> list[dict]:
    base = ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=40, patience=10, batch_size=32, neg_ratio=10, eval_every=2)
    return [
        config_row("baseline", "base", base),
        config_row("hidden", "64", ModelConfig(hidden=64, dropout=0.3, lr=5e-3, epochs=40, patience=10, batch_size=32, neg_ratio=10, eval_every=2)),
        config_row("dropout", "0.2", ModelConfig(hidden=32, dropout=0.2, lr=5e-3, epochs=40, patience=10, batch_size=32, neg_ratio=10, eval_every=2)),
        config_row("dropout", "0.4", ModelConfig(hidden=32, dropout=0.4, lr=5e-3, epochs=40, patience=10, batch_size=32, neg_ratio=10, eval_every=2)),
        config_row("lr", "1e-3", ModelConfig(hidden=32, dropout=0.3, lr=1e-3, epochs=40, patience=10, batch_size=32, neg_ratio=10, eval_every=2)),
        config_row("lr", "3e-3", ModelConfig(hidden=32, dropout=0.3, lr=3e-3, epochs=40, patience=10, batch_size=32, neg_ratio=10, eval_every=2)),
        config_row("schedule", "60_12", ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=60, patience=12, batch_size=32, neg_ratio=10, eval_every=2)),
        config_row("schedule", "80_16", ModelConfig(hidden=32, dropout=0.3, lr=5e-3, epochs=80, patience=16, batch_size=32, neg_ratio=10, eval_every=2)),
    ]


def fold_path(tag: str, fold_id: int) -> Path:
    return FOLD_DIR / f"{tag}_fold{fold_id}.pkl"


def load_config_results(tag: str) -> list[dict]:
    results = []
    for fold_id in range(10):
        path = fold_path(tag, fold_id)
        if path.exists():
            results.append(load_pickle(path))
    return sorted(results, key=lambda item: int(item["fold"]))


def maybe_bootstrap_baseline(tag: str):
    existing = load_config_results(tag)
    if len(existing) == 10:
        return
    for fold_id in range(10):
        dst = fold_path(tag, fold_id)
        if dst.exists():
            continue
        src = BASELINE_SOURCE / f"{PRIMARY_MODEL}_fold{fold_id}.pkl"
        if not src.exists():
            continue
        result = load_pickle(src)
        save_pickle(result, dst)


def run_config(cfg_row: dict, samples, labels, fold_splits):
    tag = cfg_row["tag"]
    cfg = cfg_row["config"]
    if cfg_row["axis"] == "baseline":
        maybe_bootstrap_baseline(tag)
    existing_folds = {int(r["fold"]) for r in load_config_results(tag)}
    missing = [int(split["fold"]) for split in fold_splits if int(split["fold"]) not in existing_folds]
    print(f"[sensitivity] {tag} existing={len(existing_folds)} missing={missing}", flush=True)
    for split in fold_splits:
        fold_id = int(split["fold"])
        if fold_id not in missing:
            continue
        result = train_one_split(PRIMARY_MODEL, samples, labels, split, cfg, save_model=False)
        result["sensitivity_axis"] = cfg_row["axis"]
        result["sensitivity_setting"] = cfg_row["setting"]
        result["sensitivity_tag"] = tag
        save_pickle(result, fold_path(tag, fold_id))
    return load_config_results(tag)


def build_summary_table(config_rows: list[dict], model_to_results: dict[str, list[dict]]):
    baseline_tag = config_rows[0]["tag"]
    baseline_results = model_to_results[baseline_tag]
    baseline_auroc = np.asarray([r["auroc"] for r in baseline_results], dtype=float)
    baseline_auprc = np.asarray([r["auprc"] for r in baseline_results], dtype=float)

    rows = []
    auroc_pvals = {}
    auprc_pvals = {}
    for cfg_row in config_rows:
        tag = cfg_row["tag"]
        results = model_to_results[tag]
        summary = summarize_neural(results)
        row = {
            "axis": cfg_row["axis"],
            "setting": cfg_row["setting"],
            "tag": tag,
            "hidden": cfg_row["config"].hidden,
            "dropout": cfg_row["config"].dropout,
            "lr": cfg_row["config"].lr,
            "epochs": cfg_row["config"].epochs,
            "patience": cfg_row["config"].patience,
            "batch_size": cfg_row["config"].batch_size,
            "neg_ratio": cfg_row["config"].neg_ratio,
            "selection_metric": "AUROC",
            "n_folds": len(results),
            "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
            "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
            "AUROC_mean": summary["auroc_mean"],
            "AUPRC_mean": summary["auprc_mean"],
        }
        if tag == baseline_tag:
            row["p_auroc_vs_base"] = np.nan
            row["p_auprc_vs_base"] = np.nan
        else:
            cur_auroc = np.asarray([r["auroc"] for r in results], dtype=float)
            cur_auprc = np.asarray([r["auprc"] for r in results], dtype=float)
            _, p_auroc = stats.ttest_rel(baseline_auroc, cur_auroc)
            _, p_auprc = stats.ttest_rel(baseline_auprc, cur_auprc)
            row["p_auroc_vs_base"] = float(p_auroc)
            row["p_auprc_vs_base"] = float(p_auprc)
            auroc_pvals[tag] = float(p_auroc)
            auprc_pvals[tag] = float(p_auprc)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["AUROC_mean", "AUPRC_mean"], ascending=False).reset_index(drop=True)
    df.to_csv(TABLE_DIR / "doseaware_sensitivity_summary.csv", index=False)
    df[["axis", "setting", "tag", "AUROC", "AUPRC", "n_folds", "p_auroc_vs_base", "p_auprc_vs_base"]].to_csv(
        TABLE_DIR / "doseaware_sensitivity_report.csv", index=False
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default="", help="Comma-separated config tags to run. Empty means run all configs.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only build summary tables from existing fold files.")
    args = parser.parse_args()

    ensure_dirs()
    ds, df, feature_cols, X, labels, hp, ap, pf, lookups = prepare_common_inputs()
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, [PRIMARY_MODEL])
    samples = sample_map[PRIMARY_MODEL]
    configs = candidate_configs()
    selected_tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    if selected_tags:
        configs = [cfg for cfg in configs if cfg["tag"] in selected_tags]
        missing_tags = [tag for tag in selected_tags if tag not in {cfg["tag"] for cfg in configs}]
        if missing_tags:
            raise ValueError(f"Unknown tags requested: {missing_tags}")

    model_to_results = {}
    if not args.aggregate_only:
        for cfg_row in configs:
            model_to_results[cfg_row["tag"]] = run_config(cfg_row, samples, labels, ds["fold_splits"])
        if selected_tags:
            print("\n[doseaware_sensitivity] selected tag run complete", flush=True)
            for cfg_row in configs:
                print(cfg_row["tag"], len(load_config_results(cfg_row["tag"])), flush=True)
            return

    configs = candidate_configs()
    for cfg_row in configs:
        model_to_results[cfg_row["tag"]] = load_config_results(cfg_row["tag"])
    incomplete = [cfg_row["tag"] for cfg_row in configs if len(model_to_results[cfg_row["tag"]]) < 10]
    if incomplete:
        raise RuntimeError(f"Cannot aggregate yet, incomplete configs: {incomplete}")

    summary_df = build_summary_table(configs, model_to_results)
    print("\n[doseaware_sensitivity] complete", flush=True)
    print(summary_df[["axis", "setting", "tag", "AUROC", "AUPRC", "n_folds"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
