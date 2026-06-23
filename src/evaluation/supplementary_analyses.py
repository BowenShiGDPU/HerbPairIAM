"""Supplementary slice analyses for the formal benchmark.

All analyses are post-hoc: they consume only the existing
``main_benchmark/fold_results/*.pkl`` artefacts plus the dataset / phase-1
look-ups, and never re-train any model. Because of that, every sub-command is
safe to run while GPU experiments occupy the box.

Sub-commands implement the EXPERIMENT_PROTOCOL.md sections that are pure OOF
slicing:

* ``cross_database``   -- s9.2: AUROC/AUPRC by signal source (JADER / FAERS / both)
* ``adr_coverage``     -- s9.4: AUROC/AUPRC bucketed by |targets(ADR)|
* ``formula_size``     -- s9.5: AUROC/AUPRC bucketed by |herbs(formula)|
* ``feature_importance`` -- s9.6: GBT feature importance vs DoseAwareIAM
                            channel-level attribution (Spearman + bar PDF)
* ``all``              -- run every sub-command above sequentially.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/supplementary_analyses.py all

Outputs are written into the configured ``supplementary/`` and ``figures/``
directories of the active stage. Per-cell metrics fall back to NaN when the
slice has no positive (or no negative) samples to support computing AUROC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import (
    FIGURES_DIR,
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    compute_metrics,
    ensure_output_dirs,
    load_pickle,
)


sys.stdout.reconfigure(line_buffering=True)


SOURCE_ORDER = ["JADER", "FAERS", "both"]
SOURCE_GROUP_COL = "tested_source"
ADR_COVERAGE_BUCKETS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("20+", 21, 10**9),
]
FORMULA_SIZE_BUCKETS: List[Tuple[str, int, int]] = [
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-9", 7, 9),
    ("10+", 10, 10**9),
]
DEFAULT_TOP_K_FEATURES = 15


_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def _load_dataset_df() -> pd.DataFrame:
    ds = load_pickle(_OUT_DIR / "dataset.pkl")
    return ds["df"].reset_index(drop=True)


def _load_lookups() -> dict:
    return load_pickle(_OUT_DIR / "lookups.pkl")


def _load_adr_target_profiles() -> dict:
    return load_pickle(_OUT_DIR / "adr_target_profiles.pkl")


def _load_fold_results_per_model() -> Dict[str, List[dict]]:
    if not FOLD_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Fold-results dir not found: {FOLD_RESULTS_DIR}")
    by_model: Dict[str, List[dict]] = {}
    for path in sorted(FOLD_RESULTS_DIR.glob("*_fold*.pkl")):
        try:
            obj = load_pickle(path)
        except Exception:
            continue
        model = str(obj.get("model", path.stem.split("_fold")[0]))
        by_model.setdefault(model, []).append(obj)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r.get("fold", 0)))
    return by_model


def _safe_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "auroc": float("nan"), "auprc": float("nan")}
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "auroc": float("nan"), "auprc": float("nan")}
    metrics = compute_metrics(y_true, y_prob)
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "auroc": float(metrics["auroc"]), "auprc": float(metrics["auprc"])}


def _stack_oof(model_results: List[dict], df: pd.DataFrame, group_series: pd.Series) -> pd.DataFrame:
    rows = []
    for fold in model_results:
        idx = np.asarray(fold["test_indices"], dtype=int)
        y_true = np.asarray(fold["y_true"], dtype=int)
        y_prob = np.asarray(fold["y_prob"], dtype=float)
        groups = group_series.iloc[idx].to_numpy()
        for i in range(len(idx)):
            rows.append({"group": groups[i], "y_true": int(y_true[i]), "y_prob": float(y_prob[i])})
    return pd.DataFrame(rows)


def run_cross_database(
    by_model: Dict[str, List[dict]],
    df: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    if SOURCE_GROUP_COL not in df.columns:
        raise KeyError(f"dataset.pkl df has no '{SOURCE_GROUP_COL}' column")
    rows = []
    pos_only = df[df["label"] == 1].groupby(SOURCE_GROUP_COL).size().to_dict()
    universe_total = df.groupby(SOURCE_GROUP_COL).size().to_dict()
    for model_name, results in sorted(by_model.items()):
        oof = _stack_oof(results, df, df[SOURCE_GROUP_COL])
        for source in SOURCE_ORDER:
            mask = oof["group"] == source
            sub = oof[mask]
            metrics = _safe_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
            rows.append({
                "model": model_name,
                "tested_source": source,
                "n_positive_universe": int(pos_only.get(source, 0)),
                "n_universe": int(universe_total.get(source, 0)),
                **metrics,
            })
        all_metrics = _safe_metrics(oof["y_true"].to_numpy(), oof["y_prob"].to_numpy())
        rows.append({
            "model": model_name,
            "tested_source": "ALL",
            "n_positive_universe": int(sum(pos_only.values())),
            "n_universe": int(sum(universe_total.values())),
            **all_metrics,
        })
    out_df = pd.DataFrame(rows).sort_values(["model", "tested_source"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  cross_database -> {out_csv} ({len(out_df)} rows)")
    return out_df


def _bucket_for_value(value: int, buckets: List[Tuple[str, int, int]]) -> str:
    for name, lo, hi in buckets:
        if lo <= value <= hi:
            return name
    return "other"


def run_adr_target_coverage(
    by_model: Dict[str, List[dict]],
    df: pd.DataFrame,
    adr_profiles: dict,
    out_csv: Path,
) -> pd.DataFrame:
    adr_target_count = {a: int((np.asarray(prof) > 0).sum()) for a, prof in adr_profiles.items()}
    df_local = df.copy()
    df_local["_adr_target_count"] = df_local["Adr_id"].map(adr_target_count).fillna(0).astype(int)
    df_local["_adr_bucket"] = df_local["_adr_target_count"].apply(lambda v: _bucket_for_value(int(v), ADR_COVERAGE_BUCKETS))
    rows = []
    for model_name, results in sorted(by_model.items()):
        oof = _stack_oof(results, df_local, df_local["_adr_bucket"])
        for bucket_name, _, _ in ADR_COVERAGE_BUCKETS:
            sub = oof[oof["group"] == bucket_name]
            metrics = _safe_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
            rows.append({
                "model": model_name,
                "adr_target_bucket": bucket_name,
                **metrics,
            })
    out_df = pd.DataFrame(rows).sort_values(["model", "adr_target_bucket"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  adr_target_coverage -> {out_csv} ({len(out_df)} rows)")
    return out_df


def run_formula_size(
    by_model: Dict[str, List[dict]],
    df: pd.DataFrame,
    lookups: dict,
    out_csv: Path,
) -> pd.DataFrame:
    f2h = lookups["f2h"]
    formula_size = {f: int(len(set(herbs))) for f, herbs in f2h.items()}
    df_local = df.copy()
    df_local["_n_herbs"] = df_local["TCMF_id"].map(formula_size).fillna(0).astype(int)
    df_local["_size_bucket"] = df_local["_n_herbs"].apply(lambda v: _bucket_for_value(int(v), FORMULA_SIZE_BUCKETS))
    rows = []
    for model_name, results in sorted(by_model.items()):
        oof = _stack_oof(results, df_local, df_local["_size_bucket"])
        for bucket_name, _, _ in FORMULA_SIZE_BUCKETS:
            sub = oof[oof["group"] == bucket_name]
            metrics = _safe_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
            rows.append({
                "model": model_name,
                "formula_size_bucket": bucket_name,
                **metrics,
            })
    out_df = pd.DataFrame(rows).sort_values(["model", "formula_size_bucket"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  formula_size -> {out_csv} ({len(out_df)} rows)")
    return out_df


def _xgboost_fold_importances(feature_cols: List[str]) -> np.ndarray | None:
    """Refit an XGBoost on fold 0 to recover gain-style feature importances."""

    try:
        from phase4_evaluation import default_best_configs, prepare_common_inputs
        from tabular_models import balance_train_indices, make_model
    except Exception as exc:
        print(f"  WARN: cannot refit XGBoost for importance ({exc}); skipping XGB importance")
        return None

    ds, df, _, X, labels, _, _, _, _ = prepare_common_inputs()
    split = ds["fold_splits"][0]
    train_idx = np.asarray(split["train_idx"], dtype=int)
    balanced = balance_train_indices(labels, train_idx, neg_ratio=10, seed=42)
    params = default_best_configs().get("XGBoost") or {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}
    model = make_model("XGBoost", params)
    model.fit(X[balanced], labels[balanced])
    importances = np.asarray(model.feature_importances_, dtype=float)
    return importances


def _doseaware_channel_attribution(feature_cols: List[str]) -> Dict[str, float]:
    """Aggregate the precomputed feature columns into the channel groups that
    DoseAwareIAM consumes (dose, pathway, tissue, ppi, convergence,
    complementarity, union_coverage, individual). The score is the absolute
    correlation of the column with positive-label probability mass on the
    pooled OOF predictions of DoseAwareIAM."""

    fold_dir = FOLD_RESULTS_DIR
    pkls = sorted(fold_dir.glob("DoseAwareIAM_fold*.pkl"))
    if not pkls:
        return {}
    df = _load_dataset_df()
    rows = []
    for path in pkls:
        fr = load_pickle(path)
        idx = np.asarray(fr["test_indices"], dtype=int)
        y_prob = np.asarray(fr["y_prob"], dtype=float)
        for j, sample_idx in enumerate(idx):
            rows.append({"sample_idx": int(sample_idx), "y_prob": float(y_prob[j])})
    oof = pd.DataFrame(rows).drop_duplicates("sample_idx", keep="last")
    merged = df.merge(oof, left_index=True, right_on="sample_idx", how="inner")
    contributions: Dict[str, float] = {}
    for col in feature_cols:
        x = merged[col].to_numpy(dtype=float)
        if not np.isfinite(x).all() or x.std() == 0.0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, merged["y_prob"].to_numpy(dtype=float))[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        contributions[col] = abs(corr)
    return contributions


def run_feature_importance(
    feature_cols: List[str],
    out_csv: Path,
    out_pdf: Path,
    top_k: int = DEFAULT_TOP_K_FEATURES,
) -> pd.DataFrame:
    xgb_importance = _xgboost_fold_importances(feature_cols)
    dose_channel = _doseaware_channel_attribution(feature_cols)
    rows = []
    for i, col in enumerate(feature_cols):
        rows.append({
            "feature": col,
            "xgb_importance": float(xgb_importance[i]) if xgb_importance is not None else float("nan"),
            "doseaware_abs_correlation": float(dose_channel.get(col, 0.0)),
        })
    table = pd.DataFrame(rows)
    if xgb_importance is not None:
        table["xgb_rank"] = table["xgb_importance"].rank(ascending=False, method="min").astype(int)
    table["dose_rank"] = table["doseaware_abs_correlation"].rank(ascending=False, method="min").astype(int)

    rho_pearson = float("nan")
    rho_spearman = float("nan")
    if xgb_importance is not None:
        valid = (table["xgb_importance"].abs() + table["doseaware_abs_correlation"].abs()) > 0
        if valid.sum() >= 3:
            rho_pearson = float(stats.pearsonr(table.loc[valid, "xgb_importance"], table.loc[valid, "doseaware_abs_correlation"])[0])
            rho_spearman = float(stats.spearmanr(table.loc[valid, "xgb_importance"], table.loc[valid, "doseaware_abs_correlation"])[0])
    table.attrs["pearson_rho"] = rho_pearson
    table.attrs["spearman_rho"] = rho_spearman

    table.to_csv(out_csv, index=False)
    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    pd.DataFrame([{
        "n_features": int(len(feature_cols)),
        "top_k": int(top_k),
        "pearson_rho": rho_pearson,
        "spearman_rho": rho_spearman,
    }]).to_csv(summary_path, index=False)
    print(f"  feature_importance -> {out_csv} ({len(table)} rows); summary -> {summary_path}")

    if xgb_importance is not None and table["xgb_importance"].notna().any():
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top_xgb = table.nlargest(top_k, "xgb_importance")
            top_dose = table.nlargest(top_k, "doseaware_abs_correlation")
            common = sorted(set(top_xgb["feature"]) | set(top_dose["feature"]))
            sub = table[table["feature"].isin(common)].set_index("feature").loc[common]
            fig, ax = plt.subplots(figsize=(8, max(4.0, 0.32 * len(common))))
            x = np.arange(len(common))
            width = 0.4
            ax.barh(x - width / 2, sub["xgb_importance"], height=width, label="XGBoost importance")
            ax.barh(
                x + width / 2,
                sub["doseaware_abs_correlation"] / max(sub["doseaware_abs_correlation"].max(), 1e-9) * sub["xgb_importance"].max(),
                height=width,
                label="DoseAwareIAM |corr| (rescaled)",
            )
            ax.set_yticks(x)
            ax.set_yticklabels(common, fontsize=7)
            ax.invert_yaxis()
            ax.legend(loc="lower right", fontsize=8)
            ax.set_title(f"Feature importance: XGBoost vs DoseAwareIAM channels (Spearman={rho_spearman:.3f})")
            ax.set_xlabel("importance")
            fig.tight_layout()
            fig.savefig(out_pdf)
            plt.close(fig)
            print(f"  feature_importance figure -> {out_pdf}")
        except Exception as exc:
            print(f"  WARN: skipped feature_importance.pdf ({exc})")
    return table


def run_all(args: argparse.Namespace) -> int:
    ensure_output_dirs()
    df = _load_dataset_df()
    by_model = _load_fold_results_per_model()
    if not by_model:
        print(f"No fold results under {FOLD_RESULTS_DIR}, nothing to slice.")
        return 1
    print(f"Found models: {sorted(by_model)}")
    print(f"Writing supplementary outputs to {SUPP_DIR}")

    if args.subcommand in {"cross_database", "all"}:
        run_cross_database(by_model, df, SUPP_DIR / "cross_database_analysis.csv")

    if args.subcommand in {"adr_coverage", "all"}:
        adr_profiles = _load_adr_target_profiles()
        run_adr_target_coverage(by_model, df, adr_profiles, SUPP_DIR / "adr_target_coverage_analysis.csv")

    if args.subcommand in {"formula_size", "all"}:
        lookups = _load_lookups()
        run_formula_size(by_model, df, lookups, SUPP_DIR / "formula_size_analysis.csv")

    if args.subcommand in {"feature_importance", "all"}:
        feature_cols = [c for c in df.columns if c not in {
            "TCMF_id", "Adr_id", "label", "tested_jader", "tested_faers",
            "tested_source", "signal_source", "tested_pair_count", "sample_id",
        }]
        run_feature_importance(
            feature_cols,
            SUPP_DIR / "feature_importance_consistency.csv",
            FIGURES_DIR / "feature_importance.pdf",
            top_k=args.top_k,
        )

    print("Supplementary analyses complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subcommand",
        choices=["cross_database", "adr_coverage", "formula_size", "feature_importance", "all"],
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_FEATURES, help="Top-K features for the importance bar chart.")
    args = parser.parse_args()
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
