"""
Summarize existing fold artifacts into publication tables.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import (
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    TABLES_DIR,
    calibration_table,
    compute_metrics,
    compute_pooled_predictions,
    delong_test,
    ensure_output_dirs,
    format_metric,
    holm_bonferroni,
    load_pickle,
    paired_ttest_matrix,
)


ROOT = Path(__file__).resolve().parent.parent.parent
FOLD_DIR = FOLD_RESULTS_DIR
CHAMPION_PATH = SUPP_DIR / "champion_selection.pkl"


# Model groups used to partition rows of the main benchmark table.
#   A  = tabular baselines
#   B  = heterogeneous-graph baselines
#   D  = neural set-interaction family (primary + closely-related variants)
#   Ablation = architectural ablations, reported in the ablation table only
MODEL_GROUPS = {
    # Tabular baselines
    "LogisticRegression": "A",
    "RandomForest": "A",
    "GradientBoosting": "A",
    "MLP": "A",
    "XGBoost": "A",
    # Graph baselines
    "R-GCN": "B",
    "HGT": "B",
    # Neural set-interaction family
    "HerbPairIAM": "D",           # primary model
    "IAM_Wide": "D",               # capacity-matched IAM baseline
    "InteractionAwareSetModel": "D",
    "DoseAwareIAM": "D",           # with-dose reference baseline
    "KGEmbedIAM": "D",
    "IngredientLiteIAM": "D",
    # Architectural ablations
    "HerbInteractionGraph": "Ablation",
    "AttentionPool_NoMP": "Ablation",
    "SumPool": "Ablation",
    "MeanPool": "Ablation",
    "NoADRConditioning": "Ablation",
    "DotScorer": "Ablation",
    "TwoLayerMP": "Ablation",
}


def load_fold_results():
    """Load every ``<MODEL>_fold<K>.pkl`` under FOLD_DIR, keyed by the
    filename's MODEL prefix (not by ``obj["model"]``).

    Why not ``obj["model"]``? Hard-linked fold pickles that were trained
    under an interim model name (e.g. ``DoseAware_ZeroDose``) keep the
    training-time name in the pickle payload for provenance, but the
    filename reflects the final naming (``HerbPairIAM``). Keying on the
    filename lets the aggregation match whatever name the experiment
    currently advertises while the pickle payload still documents the
    provenance internally.
    """
    import re
    _FOLD_RE = re.compile(r"^(?P<model>.+)_fold\d+$")
    model_to_results: dict[str, list[dict]] = {}
    for path in sorted(FOLD_DIR.glob("*_fold*.pkl")):
        m = _FOLD_RE.match(path.stem)
        if not m:
            continue
        model_key = m.group("model")
        try:
            obj = load_pickle(path)
        except Exception:
            continue
        # Annotate provenance so nothing is silently lost: the pickle's
        # original ``model`` field becomes ``trained_as`` when it diverges
        # from the file-level name.
        trained_as = obj.get("model")
        if trained_as and trained_as != model_key:
            obj.setdefault("trained_as", trained_as)
        obj["model"] = model_key
        model_to_results.setdefault(model_key, []).append(obj)
    for model in model_to_results:
        model_to_results[model] = sorted(model_to_results[model], key=lambda r: int(r.get("fold", 0)))
    return model_to_results


def summarize_model(results):
    out = {}
    for key in ["auroc", "auprc", "precision", "recall", "f1", "mcc"]:
        vals = np.asarray([float(r[key]) for r in results], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=0))
    return out


def main():
    ensure_output_dirs()
    # Champion model: prefer the artefact if it exists and points to a model
    # present in the current fold results; otherwise fall back to the canonical
    # PRIMARY_MODEL_NAME. This keeps backward compatibility with old runs
    # while defaulting to the current primary (``HerbPairIAM``) for new runs.
    try:
        from phase4_evaluation import PRIMARY_MODEL_NAME
    except Exception:
        PRIMARY_MODEL_NAME = "HerbPairIAM"

    model_to_results = load_fold_results()
    champion = PRIMARY_MODEL_NAME
    if CHAMPION_PATH.exists():
        try:
            artefact = load_pickle(CHAMPION_PATH)
            candidate = artefact.get("champion") if isinstance(artefact, dict) else None
            # Only honour the artefact if it names a model we actually have folds for.
            if candidate and candidate in model_to_results:
                champion = candidate
        except Exception:
            pass
    if champion not in model_to_results:
        print(f"[warn] champion={champion!r} has no fold pkls; skipping pooled stats.")
        champion = None

    complete_models = {m: rs for m, rs in model_to_results.items() if len(rs) == 10 and MODEL_GROUPS.get(m) in {"A", "B", "C", "D"}}
    partial_models = {m: rs for m, rs in model_to_results.items() if len(rs) != 10}

    rows = []
    for model, results in complete_models.items():
        summary = summarize_model(results)
        rows.append(
            {
                "Model": model,
                "Group": MODEL_GROUPS.get(model, ""),
                "AUROC": format_metric(summary["auroc_mean"], summary["auroc_std"]),
                "AUPRC": format_metric(summary["auprc_mean"], summary["auprc_std"]),
                "AUROC_mean": summary["auroc_mean"],
                "AUPRC_mean": summary["auprc_mean"],
                "n_folds": len(results),
            }
        )
    benchmark_df = pd.DataFrame(rows).sort_values(["AUPRC_mean", "AUROC_mean"], ascending=False).reset_index(drop=True)

    if champion is not None and champion in complete_models:
        pvals = {}
        champion_scores = np.asarray([r["auprc"] for r in complete_models[champion]], dtype=float)
        for model, results in complete_models.items():
            if model == champion:
                continue
            rhs = np.asarray([r["auprc"] for r in results], dtype=float)
            _, pvalue = stats.ttest_rel(champion_scores, rhs)
            pvals[model] = float(pvalue)
        benchmark_df["pHolm_AUPRC_vs_champion"] = benchmark_df["Model"].map(holm_bonferroni(pvals))
        paired_ttest_matrix(complete_models, "auroc").to_csv(TABLES_DIR / "significance_matrix_auroc.csv")
        paired_ttest_matrix(complete_models, "auprc").to_csv(TABLES_DIR / "significance_matrix_auprc.csv")

        champion_true, champion_prob = compute_pooled_predictions(complete_models[champion])
        pooled_rows = []
        for model, results in complete_models.items():
            y_true, y_prob = compute_pooled_predictions(results)
            pooled_rows.append(
                {
                    "Model": model,
                    "n": len(y_true),
                    "pooled_auroc": compute_metrics(y_true, y_prob)["auroc"],
                    "pooled_auprc": compute_metrics(y_true, y_prob)["auprc"],
                    "delong_p_vs_champion": np.nan if model == champion else delong_test(champion_true, champion_prob, y_prob),
                }
            )
        pd.DataFrame(pooled_rows).to_csv(SUPP_DIR / "pooled_significance.csv", index=False)

        calib = calibration_table(champion_true, champion_prob)
        calib["model"] = champion
        calib.to_csv(SUPP_DIR / "calibration.csv", index=False)

    benchmark_df.to_csv(TABLES_DIR / "main_benchmark.csv", index=False)

    if partial_models:
        partial_rows = []
        for model, results in partial_models.items():
            summary = summarize_model(results)
            partial_rows.append(
                {
                    "Model": model,
                    "n_folds": len(results),
                    "AUROC_mean": summary["auroc_mean"],
                    "AUPRC_mean": summary["auprc_mean"],
                }
            )
        pd.DataFrame(partial_rows).sort_values(["n_folds", "AUPRC_mean"], ascending=[True, False]).to_csv(
            SUPP_DIR / "incomplete_models_progress.csv", index=False
        )

    print("Wrote:")
    print(TABLES_DIR / "main_benchmark.csv")
    if (TABLES_DIR / "significance_matrix_auroc.csv").exists():
        print(TABLES_DIR / "significance_matrix_auroc.csv")
        print(TABLES_DIR / "significance_matrix_auprc.csv")
        print(SUPP_DIR / "pooled_significance.csv")
        print(SUPP_DIR / "calibration.csv")


if __name__ == "__main__":
    main()
