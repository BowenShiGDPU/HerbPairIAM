"""Render the main publication figures from the existing artefacts.

This script is read-only: it never trains. It sweeps the active stage's
``fold_results/`` for ROC / PR curves, looks at
``supplementary/calibration.csv`` (or rebuilds calibration from pooled
predictions), reads ``tables/cold_start.csv`` if present, and consumes the
parameter sensitivity summary that lives outside the formal stage tree.

Outputs land in ``<stage>/figures/``:

* ``roc_curve_main.pdf``
* ``pr_curve_main.pdf``
* ``calibration_diagram.pdf``
* ``cold_start_comparison.pdf``        (if cold_start.csv is available)
* ``parameter_sensitivity.pdf``        (if sensitivity summary is available)
* ``feature_importance.pdf``           (if feature_importance_consistency.csv is present)
* ``attention_heatmap_example.pdf``    (uses the case_*.json files written by phase5)

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/make_figures.py [--include FIG,FIG]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve

from experiment_utils import (
    FIGURES_DIR,
    FOLD_RESULTS_DIR,
    INTERPRET_DIR,
    SUPP_DIR,
    TABLES_DIR,
    calibration_table,
    compute_pooled_predictions,
    ensure_output_dirs,
    load_pickle,
)


sys.stdout.reconfigure(line_buffering=True)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SENSITIVITY_SUMMARY = ROOT_DIR / "outputs" / "doseaware_sensitivity_val_auroc" / "tables" / "doseaware_sensitivity_summary.csv"
PRIMARY_MODEL = "DoseAwareIAM"
PLOT_STYLE = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "font.family": "sans-serif",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _apply_style():
    matplotlib.rcParams.update(PLOT_STYLE)


def _load_pooled_by_model() -> Dict[str, dict]:
    if not FOLD_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Fold results dir not found: {FOLD_RESULTS_DIR}")
    by_model: Dict[str, list] = {}
    for path in sorted(FOLD_RESULTS_DIR.glob("*_fold*.pkl")):
        try:
            obj = load_pickle(path)
        except Exception:
            continue
        if "y_true" not in obj or "y_prob" not in obj:
            continue
        model = str(obj.get("model", path.stem.split("_fold")[0]))
        by_model.setdefault(model, []).append(obj)
    pooled: Dict[str, dict] = {}
    for model, results in by_model.items():
        results.sort(key=lambda r: int(r.get("fold", 0)))
        y_true, y_prob = compute_pooled_predictions(results)
        pooled[model] = {"y_true": y_true, "y_prob": y_prob, "n_folds": len(results)}
    return pooled


def fig_roc_main(pooled: Dict[str, dict], out_pdf: Path):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for model, data in sorted(pooled.items()):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y_true, y_prob))
        line, = ax.plot(fpr, tpr, lw=1.4, label=f"{model} (AUROC={auroc:.3f})")
        if model == PRIMARY_MODEL:
            line.set_lw(2.2)
            line.set_zorder(10)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle="--", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Pooled OOF ROC across 10 folds")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_pr_main(pooled: Dict[str, dict], out_pdf: Path):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for model, data in sorted(pooled.items()):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score

        auprc = float(average_precision_score(y_true, y_prob))
        line, = ax.plot(recall, precision, lw=1.4, label=f"{model} (AUPRC={auprc:.3f})")
        if model == PRIMARY_MODEL:
            line.set_lw(2.2)
            line.set_zorder(10)
    base_rate = float(pooled[PRIMARY_MODEL]["y_true"].mean()) if PRIMARY_MODEL in pooled else 0.0
    if base_rate > 0:
        ax.axhline(base_rate, color="grey", lw=0.8, linestyle="--", label=f"Base rate={base_rate:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pooled OOF Precision-Recall across 10 folds")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_calibration(pooled: Dict[str, dict], out_pdf: Path):
    if PRIMARY_MODEL not in pooled:
        print(f"  WARN: {PRIMARY_MODEL} pooled predictions missing, skip calibration figure.")
        return
    y_true = pooled[PRIMARY_MODEL]["y_true"]
    y_prob = pooled[PRIMARY_MODEL]["y_prob"]
    if y_true.size == 0:
        return
    table = calibration_table(y_true, y_prob, n_bins=10)
    table = table.dropna(subset=["mean_pred", "empirical_pos_rate"])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], lw=0.8, color="grey", linestyle="--", label="Perfect calibration")
    ax.plot(table["mean_pred"], table["empirical_pos_rate"], marker="o", lw=1.8, label=PRIMARY_MODEL)
    for _, row in table.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            (float(row["mean_pred"]), float(row["empirical_pos_rate"])),
            textcoords="offset points",
            xytext=(4, -8),
            fontsize=6,
            color="dimgrey",
        )
    ece = float(table["ece"].iloc[0]) if "ece" in table.columns and not table.empty else float("nan")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"DoseAwareIAM calibration (ECE={ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_cold_start(out_pdf: Path):
    csv = TABLES_DIR / "cold_start.csv"
    if not csv.exists():
        print(f"  cold_start.csv not found at {csv}, skipping cold-start figure.")
        return
    df = pd.read_csv(csv)
    if df.empty:
        print("  cold_start.csv is empty, skipping figure.")
        return
    long_df = df.copy()
    metric_col = None
    for cand in ["AUROC_mean", "auroc", "auroc_mean", "AUROC"]:
        if cand in long_df.columns:
            metric_col = cand
            break
    if metric_col is None:
        print("  Could not find AUROC column in cold_start.csv, skipping cold-start figure.")
        return
    grouping_col = None
    for cand in ["split_type", "split", "Split", "cold_start"]:
        if cand in long_df.columns:
            grouping_col = cand
            break
    if grouping_col is None:
        long_df["_split"] = "ALL"
        grouping_col = "_split"
    if "model" in long_df.columns:
        long_df = long_df.rename(columns={"model": "Model"})

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    splits = sorted(long_df[grouping_col].unique())
    models = sorted(long_df["Model"].unique())
    bar_width = 0.8 / max(len(models), 1)
    x_base = np.arange(len(splits), dtype=float)
    for i, model in enumerate(models):
        sub = long_df[long_df["Model"] == model].set_index(grouping_col).reindex(splits)
        means = sub[metric_col].to_numpy(dtype=float)
        std_col = metric_col.replace("mean", "std").replace("AUROC", "AUROC_std") if "_mean" in metric_col else None
        std = sub[std_col].to_numpy(dtype=float) if std_col and std_col in sub.columns else np.zeros_like(means)
        ax.bar(x_base + i * bar_width - 0.4 + bar_width / 2, means, width=bar_width, yerr=std, capsize=2, label=model)
    ax.set_xticks(x_base)
    ax.set_xticklabels(splits)
    ax.set_ylabel("AUROC")
    ax.set_title("Cold-start AUROC by model")
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_parameter_sensitivity(out_pdf: Path):
    if not SENSITIVITY_SUMMARY.exists():
        print(f"  Parameter sensitivity summary not found at {SENSITIVITY_SUMMARY}; skip figure.")
        return
    df = pd.read_csv(SENSITIVITY_SUMMARY)
    if df.empty:
        return
    axes_to_plot = ["lr", "hidden", "dropout", "schedule"]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True)
    for ax, axis_key in zip(axes, axes_to_plot):
        sub = df[df["axis"].isin([axis_key, "baseline"])].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("setting")
        ax.errorbar(
            sub["setting"].astype(str),
            sub["AUROC_mean"],
            yerr=sub["AUROC_mean"].apply(lambda _: 0.0),
            marker="o",
            lw=1.2,
        )
        for _, row in sub.iterrows():
            label = row["setting"]
            ax.annotate(label, (row["setting"], row["AUROC_mean"]), textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
        ax.set_title(f"axis = {axis_key}")
        ax.set_xlabel(axis_key)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("AUROC (10-fold mean)")
    fig.suptitle("DoseAwareIAM parameter sensitivity")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_feature_importance_link(out_pdf: Path):
    csv = SUPP_DIR / "feature_importance_consistency.csv"
    pdf_target = FIGURES_DIR / "feature_importance.pdf"
    if not csv.exists():
        print(f"  Skipping feature_importance figure ({csv} missing).")
        return
    if pdf_target.exists():
        print(f"  feature_importance.pdf already produced by supplementary_analyses ({pdf_target}); reuse.")
        return
    df = pd.read_csv(csv).sort_values("xgb_importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.4
    x = np.arange(len(df))
    ax.barh(x - width / 2, df["xgb_importance"], height=width, label="XGBoost importance")
    if df["doseaware_abs_correlation"].max() > 0:
        scaled = df["doseaware_abs_correlation"] / df["doseaware_abs_correlation"].max() * df["xgb_importance"].max()
    else:
        scaled = df["doseaware_abs_correlation"]
    ax.barh(x + width / 2, scaled, height=width, label="DoseAwareIAM |corr| (rescaled)")
    ax.set_yticks(x)
    ax.set_yticklabels(df["feature"], fontsize=7)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.set_title("Feature importance: XGBoost vs DoseAwareIAM channels")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_attention_heatmap(out_pdf: Path):
    case_files = sorted(INTERPRET_DIR.glob("case_*.json"))
    if not case_files:
        print("  No case_*.json files found, skip attention heatmap.")
        return
    chosen = case_files[: min(3, len(case_files))]
    cases = []
    for path in chosen:
        with open(path, "r", encoding="utf-8") as f:
            cases.append(json.load(f))
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        herbs = [entry["herb_name"] for entry in case["herb_attention_ranked"]]
        herb_attn = np.asarray([entry["attention"] for entry in case["herb_attention_ranked"]], dtype=float)
        order = np.argsort(-herb_attn)
        herbs = [herbs[i] for i in order]
        herb_attn = herb_attn[order]

        pair_records = case.get("top_pair_interactions", [])
        pair_labels = [f"{p['h1_name'][:4]} x {p['h2_name'][:4]}" for p in pair_records[:5]]
        pair_scores = np.asarray([p.get("interaction_score", 0.0) for p in pair_records[:5]], dtype=float)

        block = max(len(herbs), len(pair_labels), 1)
        matrix = np.full((block, 2), np.nan, dtype=float)
        matrix[: len(herbs), 0] = herb_attn
        matrix[: len(pair_scores), 1] = pair_scores

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["herb attn", "pair score"], fontsize=8)
        labels = []
        for i in range(block):
            left = herbs[i] if i < len(herbs) else ""
            right = pair_labels[i] if i < len(pair_labels) else ""
            labels.append(f"{left} | {right}")
        ax.set_yticks(np.arange(block))
        ax.set_yticklabels(labels, fontsize=6)
        formula_label = case.get("TCMF_id", "?")
        ax.set_title(
            f"{formula_label} -> {case.get('ADR_name', case['Adr_id'])}\nprob={case['predicted_probability']:.3f}",
            fontsize=8,
        )
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


FIGURES = {
    "roc": ("roc_curve_main.pdf", lambda pooled: fig_roc_main(pooled, FIGURES_DIR / "roc_curve_main.pdf")),
    "pr": ("pr_curve_main.pdf", lambda pooled: fig_pr_main(pooled, FIGURES_DIR / "pr_curve_main.pdf")),
    "calibration": ("calibration_diagram.pdf", lambda pooled: fig_calibration(pooled, FIGURES_DIR / "calibration_diagram.pdf")),
    "cold_start": ("cold_start_comparison.pdf", lambda pooled: fig_cold_start(FIGURES_DIR / "cold_start_comparison.pdf")),
    "parameter_sensitivity": ("parameter_sensitivity.pdf", lambda pooled: fig_parameter_sensitivity(FIGURES_DIR / "parameter_sensitivity.pdf")),
    "feature_importance": ("feature_importance.pdf", lambda pooled: fig_feature_importance_link(FIGURES_DIR / "feature_importance.pdf")),
    "attention_heatmap": ("attention_heatmap_example.pdf", lambda pooled: fig_attention_heatmap(FIGURES_DIR / "attention_heatmap_example.pdf")),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include",
        default="",
        help="Comma-separated subset of figures (default: all)",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    _apply_style()

    selection = list(FIGURES.keys()) if not args.include else [s.strip() for s in args.include.split(",") if s.strip()]
    print(f"Rendering figures into {FIGURES_DIR}: {selection}")
    pooled = _load_pooled_by_model() if FOLD_RESULTS_DIR.exists() else {}
    for key in selection:
        if key not in FIGURES:
            print(f"  WARN: unknown figure '{key}', skipped.")
            continue
        out_name, fn = FIGURES[key]
        print(f"  -> {out_name}")
        try:
            fn(pooled)
        except Exception as exc:
            print(f"     FAILED: {exc}")
    print("Figure rendering complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
