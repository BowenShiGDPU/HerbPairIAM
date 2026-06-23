"""Polish every CSV in paper_package/ into a top-journal-ready artefact.

Operates on the already-assembled ``paper_package/`` tree and performs
four orthogonal polish steps:

1. **File reorganisation** (`_RENAMES`). Implements the
   ``FIGURE_TABLE_PLAN.md`` routing — promotes primary-vs-baselines
   from supplementary to Main Table 3, demotes alliance / cold-start /
   HerbEmb tables from main to supplementary, renumbers every
   enumerated supplementary table to its planned ``Sn`` slot, and
   moves subsumed / legacy artefacts under
   ``paper_package/provenance/tables/``.

2. **Column-name normalisation** (`_rename_columns`). Everything is
   lowered to ``snake_case`` so the CSVs match the rest of the
   data-engineering codebase. Domain-standard acronyms stay inside
   the snake_case keys (``auroc_mean`` not ``AUROC_mean``); the
   published figures still use ``AUROC`` / ``AUPRC`` in axis labels
   and captions — that is a rendering concern, not a data concern.

3. **Numeric precision normalisation** (`_round_numeric_columns`).
   Metric columns are rounded to 4 decimals, p-values to three
   significant figures in scientific notation, Spearman ρ to three
   decimals, and integer-like counts stay integers. This enforces
   the same precision a reviewer sees in the paper's tables and
   removes the 15-decimal raw floats that look like unpolished
   intermediate artefacts.

4. **Content fixes.** Adds a ``formula_romanised`` column to
   ``supp_table_case_summary.csv`` so no kanji appears in a cited
   CSV. Drops the redundant "mean±std" string columns in the
   canonical-seed-42 benchmark since the numeric mean and std columns
   already carry that information.

Run from the repo root::

    python src/scripts/polish_tables.py

The script is idempotent: running it on an already-polished tree is a
no-op. It requires only the files produced by
``src/scripts/assemble_paper_package.py``; run that first on a fresh
checkout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
PKG  = ROOT / "paper_package"

# ---------------------------------------------------------------------------
# 1. File-reorganisation plan.  ``(src_subdir, filename) -> (dst_subdir, filename)``
# ---------------------------------------------------------------------------
_RENAMES: dict[tuple[str, str], tuple[str, str]] = {
    # Main Table 1: structure ablation (was table2)
    ("main/tables", "table2_structure_ablation.csv"):
        ("main/tables", "table1_structure_ablation.csv"),
    # Main Table 2: feature ablation (was table3)
    ("main/tables", "table3_feature_ablation.csv"):
        ("main/tables", "table2_feature_ablation.csv"),
    # Main Table 3: primary vs baselines paired 30-fold (promoted from supp)
    ("supplementary", "supp_table_primary_vs_baselines_paired_30fold.csv"):
        ("main/tables", "table3_primary_vs_baselines.csv"),

    # Demote (were Tables 4/5/6) → Supp Tables S3/S4/S5
    ("main/tables", "table4_alliance_ablation.csv"):
        ("supplementary", "supp_table_S3_alliance_ablation.csv"),
    ("main/tables", "table5_cold_start.csv"):
        ("supplementary", "supp_table_S4_cold_start.csv"),
    ("main/tables", "table6_herbpair_vs_herbemb_paired.csv"):
        ("supplementary", "supp_table_S5_herbpair_vs_herbemb.csv"),

    # Enumerate the remaining Supp Tables (S1, S2, S6, S7, S8)
    ("supplementary", "supp_table_main_benchmark_per_seed.csv"):
        ("supplementary", "supp_table_S1_main_benchmark_per_seed.csv"),
    ("supplementary", "supp_table_delong_holm.csv"):
        ("supplementary", "supp_table_S2_delong_holm.csv"),
    ("supplementary", "supp_table_dose_head2head_pooled.csv"):
        ("supplementary", "supp_table_S6_dose_head2head_pooled.csv"),
    ("supplementary", "supp_table_feature_ablation_xgboost.csv"):
        ("supplementary", "supp_table_S7_feature_ablation_xgboost.csv"),
    ("supplementary", "supp_table_feature_ablation_doseaware.csv"):
        ("supplementary", "supp_table_S8_feature_ablation_doseaware.csv"),

    # Move provenance / subsumed artefacts out of the cited namespace.
    # These are kept for data availability but not cited anywhere.
    ("main/tables", "table1_main_benchmark_canonical.csv"):
        ("provenance/tables", "main_benchmark_canonical_seed42.csv"),
    ("main/tables", "table1_main_benchmark_multiseed.csv"):
        ("provenance/tables", "main_benchmark_multiseed_underlying_fig4.csv"),
    ("supplementary", "supp_table_significance_matrix_auroc.csv"):
        ("provenance/tables", "legacy_significance_matrix_auroc.csv"),
    ("supplementary", "supp_table_significance_matrix_auprc.csv"):
        ("provenance/tables", "legacy_significance_matrix_auprc.csv"),
    ("supplementary", "supp_table_pooled_significance.csv"):
        ("provenance/tables", "legacy_pooled_significance.csv"),
    ("supplementary", "supp_table_delong_matrix.csv"):
        ("provenance/tables", "delong_pvalue_matrix.csv"),
    ("supplementary", "supp_table_bootstrap_cluster_ci.csv"):
        ("provenance/tables", "bootstrap_cluster_ci_main_benchmark.csv"),
    ("supplementary", "supp_table_structure_ablation_bootstrap_ci.csv"):
        ("provenance/tables", "bootstrap_cluster_ci_structure_ablation.csv"),
    ("supplementary", "supp_table_feature_ablation_bootstrap_ci.csv"):
        ("provenance/tables", "bootstrap_cluster_ci_feature_ablation.csv"),
    ("supplementary", "supp_table_alliance_ablation_bootstrap_ci.csv"):
        ("provenance/tables", "bootstrap_cluster_ci_alliance_ablation.csv"),
    ("supplementary", "supp_table_neg_sampling_bootstrap_ci.csv"):
        ("provenance/tables", "bootstrap_cluster_ci_neg_sampling.csv"),
    ("supplementary", "supp_table_cold_start_bootstrap_ci.csv"):
        ("provenance/tables", "bootstrap_cluster_ci_cold_start.csv"),
    ("supplementary", "supp_table_herb_attention_consistency.csv"):
        ("provenance/tables", "herb_attention_consistency.csv"),
    ("supplementary", "supp_table_top_novel_predictions.csv"):
        ("provenance/tables", "top_novel_predictions_data_availability.csv"),
    ("supplementary", "supp_table_dose_head2head_summary.csv"):
        ("provenance/tables", "dose_head2head_per_seed.csv"),
}


# ---------------------------------------------------------------------------
# 2. Column-name normalisation
# ---------------------------------------------------------------------------
# Per-column regex replacements applied in order.
_COL_REGEXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^AUROC$"),       "auroc"),
    (re.compile(r"^AUPRC$"),       "auprc"),
    (re.compile(r"AUROC"),         "auroc"),
    (re.compile(r"AUPRC"),         "auprc"),
    (re.compile(r"Adr_id"),        "adr_id"),
    (re.compile(r"ADR_name"),      "adr_name"),
    (re.compile(r"TCMF_id"),       "tcmf_id"),
    (re.compile(r"^Model$"),       "model"),
    (re.compile(r"^Setting$"),     "setting"),
    (re.compile(r"^Group$"),       "group"),
    (re.compile(r"^pHolm"),        "p_holm"),
    # Unify the three naming conventions for the champion comparison.
    (re.compile(r"_vs_champion$"), "_vs_primary"),
    (re.compile(r"_vs_full$"),     "_vs_primary"),
    (re.compile(r"_champion$"),    "_primary"),
    (re.compile(r"split_type"),    "split_type"),     # no-op; guard
]


def _snake_case(s: str) -> str:
    """Lowercase snake-case transform for a column name."""
    x = s
    for pat, repl in _COL_REGEXES:
        x = pat.sub(repl, x)
    # Insert underscores at camelCase boundaries only where no underscore
    # is already present; e.g. "HerbPairIAM_auroc_mean" → keep HerbPairIAM
    # (model-name literals) but lowercase overall segment casing.
    # We never want to touch value cells — only column names.
    x = x.strip()
    # Lowercase letters that are followed by an underscore or end-of-name.
    # But keep model-name literals (HerbPairIAM, DoseAwareIAM, …) within
    # the column name if they appear as a prefix like "HerbPairIAM_auroc_mean".
    # We simply lowercase the whole thing because the PRIMARY_LITERALS
    # below recover any canonical rendering downstream.
    return x


# Primary-model literals preserved in column *keys* (e.g. "HerbPairIAM_auroc").
# When we lowercase for CSV uniformity we keep the lowercase version
# everywhere; readers who want "HerbPairIAM" treat it as the value of a
# ``model`` column, not as part of a column key.
_MODEL_LITERAL_TOKENS = (
    "HerbPairIAM", "DoseAwareIAM", "IAM_Wide", "InteractionAwareSetModel",
    "HerbEmbIAM",
)


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``_COL_REGEXES`` then lowercase everything safely."""
    new_cols = []
    for c in df.columns:
        # Step 1: fixed regex-based substitutions (AUROC → auroc etc.)
        new = _snake_case(c)
        # Step 2: lowercase unless the column is a literal value that we
        # genuinely want preserved as-is.  All our tables keep model
        # names in a ``model`` *row*, never as part of a column *key*,
        # except the paired-comparison table which uses
        # ``HerbPairIAM_auroc_mean`` / ``baseline_auroc_mean``.  We
        # preserve that literal for clarity, only lowering the metric
        # suffix.
        if any(tok in new for tok in _MODEL_LITERAL_TOKENS):
            # leave literal capitalisation of the model token, lowercase the rest
            parts = new.split("_")
            new_parts = []
            for p in parts:
                if p in _MODEL_LITERAL_TOKENS:
                    new_parts.append(p)
                else:
                    new_parts.append(p.lower())
            new = "_".join(new_parts)
        else:
            new = new.lower()
        new_cols.append(new)
    df = df.copy()
    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# 3. Numeric precision normalisation
# ---------------------------------------------------------------------------
# Column-name suffix → rounding spec.
_METRIC_DECIMALS = 4     # auroc, auprc, f1, mcc, brier, ece, delta_*
_P_SIGFIGS       = 3     # wilcoxon_p_*, p_holm_*, delong_p_*, spearman_p
_RHO_DECIMALS    = 3     # spearman_rho
_CI_DECIMALS     = 4     # ci_low / ci_high


def _round_sigfig(x: float, sig: int) -> float:
    """Round ``x`` to ``sig`` significant figures."""
    if pd.isna(x) or x == 0:
        return x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


_METRIC_NAMES = (
    "auroc", "auprc", "f1", "mcc", "brier", "ece",
    "precision", "recall",
    "mean_predicted", "observed_frequency", "net_benefit",
    "attention", "abs_delta_prob", "delta_prob", "prob",
    "predicted_probability",
    "prob_full", "prob_without_herb",
    "jaccard",
)


def _is_p_col(name: str) -> bool:
    """A column is a p-value column if it starts with ``p_``, contains
    ``_p_`` as a delimited token, or ends with ``_p``. This catches both
    styles we see in practice: ``p_holm_auroc_vs_primary`` (name starts
    with ``p_``) and ``wilcoxon_holm_p_auroc`` (embedded ``_p_``).

    The earlier suffix-match definition (``p_holm``, ``wilcoxon_p`` …)
    missed ``wilcoxon_holm_p_auroc`` and caused the subsequent metric
    rounding to zero-out 1e-8 Holm p-values in Main Table 3.
    """
    n = name.lower()
    if n.startswith("p_"):
        return True
    if "_p_" in n:
        return True
    if n.endswith("_p"):
        return True
    return False


def _is_rho_col(name: str) -> bool:
    return "spearman_rho" in name.lower()


def _is_metric_col(name: str) -> bool:
    n = name.lower()
    if any(n == m or n.endswith("_" + m) or n.startswith(m + "_") for m in _METRIC_NAMES):
        return True
    if "delta_" in n and ("auroc" in n or "auprc" in n):
        return True
    if n.endswith("_mean") or n.endswith("_std") or n.endswith("_se"):
        # mean/std/se of a metric → metric precision
        return True
    if "ci_low" in n or "ci_high" in n:
        return True
    return False


def _round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind not in "fi":
            continue
        if _is_p_col(col):
            df[col] = df[col].apply(lambda v: _round_sigfig(v, _P_SIGFIGS)
                                    if pd.notna(v) else v)
        elif _is_rho_col(col):
            df[col] = df[col].round(_RHO_DECIMALS)
        elif _is_metric_col(col):
            df[col] = df[col].round(_METRIC_DECIMALS)
        else:
            # Integer-valued columns stored as float → cast to Int
            if df[col].dtype.kind == "f" and df[col].dropna().apply(float.is_integer).all():
                try:
                    df[col] = df[col].astype("Int64")
                except Exception:
                    pass
    return df


# ---------------------------------------------------------------------------
# 4. Content fixes
# ---------------------------------------------------------------------------
_FORMULA_ROMANISATION: dict[str, str] = {
    "小柴胡湯":       "Sho-saiko-to",
    "清上防風湯":     "Seijo-bofu-to",
    "補中益気湯":     "Hochu-ekki-to",
    "小柴胡湯加桔梗石膏": "Sho-saiko-to ka Kikyo-Sekko",
    "加味帰脾湯":     "Kami-kihi-to",
    "大黄甘草湯":     "Dai-o-kanzo-to",
    "女神散":         "Nyoshin-san",
    "防風通聖散":     "Bofu-tsusho-san",
    "黄連解毒湯":     "Oren-gedoku-to",
    "乙字湯":         "Otsuji-to",
    "柴苓湯":         "Sai-rei-to",
    "芍薬甘草湯":     "Shakuyaku-kanzo-to",
    "白虎加人参湯":   "Byakko-ka-ninjin-to",
    "麦門冬湯":       "Bakumon-do-to",
    "大建中湯":       "Dai-kenchu-to",
    "八味地黄丸":     "Hachimi-jio-gan",
    "当帰芍薬散":     "Toki-shakuyaku-san",
}


def _inject_romanisation(df: pd.DataFrame) -> pd.DataFrame:
    """If the frame has a ``formula_name`` column with kanji values,
    insert a ``formula_romanised`` column to its right."""
    if "formula_name" not in df.columns:
        return df
    if "formula_romanised" in df.columns:
        return df
    roman = df["formula_name"].map(
        lambda x: _FORMULA_ROMANISATION.get(str(x), str(x))
    )
    cols = list(df.columns)
    insert_at = cols.index("formula_name") + 1
    df = df.copy()
    df.insert(insert_at, "formula_romanised", roman)
    return df


def _drop_redundant_meanstd_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Table-1-canonical had ``AUROC='0.8245±0.0209'`` strings next to
    ``auroc_mean=0.8245``.  The string is redundant and non-numeric, so
    we drop it if both are present."""
    drop = []
    for c in list(df.columns):
        if c in ("auroc", "auprc") and df[c].dtype == object:
            # Only drop when a companion _mean column exists
            if f"{c}_mean" in df.columns:
                drop.append(c)
    if drop:
        df = df.drop(columns=drop)
    return df


# ---------------------------------------------------------------------------
# Polish pipeline (per-file)
# ---------------------------------------------------------------------------
def _polish_one(path: Path) -> None:
    """Read a single CSV, apply the four polish steps, write it back."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[polish] skip {path.relative_to(PKG)} (read failed: {exc})")
        return
    original_shape = df.shape

    df = _rename_columns(df)
    df = _drop_redundant_meanstd_strings(df)
    df = _inject_romanisation(df)
    df = _round_numeric_columns(df)

    # Write back with controlled float formatting so rounded values don't
    # re-introduce 15-decimal noise via pandas' default repr.
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"[polish] {path.relative_to(PKG)}  {original_shape} → {df.shape}")


# ---------------------------------------------------------------------------
# File-reorganisation driver
# ---------------------------------------------------------------------------
def _apply_renames() -> None:
    (PKG / "provenance" / "tables").mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for (src_dir, src_name), (dst_dir, dst_name) in _RENAMES.items():
        src_path = PKG / src_dir / src_name
        dst_path = PKG / dst_dir / dst_name
        if dst_path.exists() and not src_path.exists():
            continue                                              # already done
        if not src_path.exists():
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        src_path.rename(dst_path)
        moved_count += 1
        print(f"[polish] moved  {src_dir}/{src_name}  →  {dst_dir}/{dst_name}")
    print(f"[polish] {moved_count} file(s) reorganised.")


# ---------------------------------------------------------------------------
# TABLES_README.md generator
# ---------------------------------------------------------------------------
_MAIN_TABLES = [
    ("table1_structure_ablation.csv",
     "Main Table 1 — Architectural ablation of HerbPairIAM",
     "9 rows × 12 columns. HerbPairIAM vs eight structural variants on "
     "the canonical 10-fold benchmark. Columns: model, n_folds, auroc, "
     "auroc_std, auprc, auprc_std, delta_auroc_vs_primary, "
     "delta_auprc_vs_primary, p_auroc_vs_primary, p_auprc_vs_primary, "
     "p_holm_auroc_vs_primary, p_holm_auprc_vs_primary. "
     "p-values are paired two-sided Wilcoxon, Holm-corrected across 8 "
     "variants."),
    ("table2_feature_ablation.csv",
     "Main Table 2 — Feature-group ablation of HerbPairIAM",
     "9 rows × 13 columns. Feature-group settings (full, without_dose, "
     "without_pathway, without_tissue, without_ppi, without_convergence, "
     "without_complementarity, without_union_coverage, pair_zero_diagnostic) "
     "on the canonical 10-fold benchmark. "
     "p-values are paired two-sided Wilcoxon vs `full`, Holm-corrected."),
    ("table3_primary_vs_baselines.csv",
     "Main Table 3 — HerbPairIAM vs every externally-defined baseline "
     "on 30 paired folds (3 seeds × 10 folds)",
     "10 rows × 18 columns. For each baseline: fold-level mean AUROC and "
     "AUPRC for HerbPairIAM and the baseline, Δ and its SE, number of "
     "wins out of 30, paired two-sided Wilcoxon P, paired t-test P "
     "(reference only), and the Holm-adjusted Wilcoxon P across 10 "
     "baselines."),
]

_SUPP_TABLES = [
    ("supp_table_S1_main_benchmark_per_seed.csv",
     "Supp Table S1 — Main benchmark per-seed breakdown",
     "Per-seed, per-model 10-fold summaries for the 3 outer-CV seeds "
     "(42, 13, 7) used in the multi-seed benchmark."),
    ("supp_table_S2_delong_holm.csv",
     "Supp Table S2 — Pooled-OOF DeLong pairwise comparisons",
     "All-pairs DeLong test for the pooled-OOF AUROC of every model, "
     "Holm-corrected across the matrix."),
    ("supp_table_S3_alliance_ablation.csv",
     "Supp Table S3 — Alliance-level feature ablation (5 rows)",
     "Groups of feature families turned off together "
     "(AL_individual, AL_pair_direct, AL_pair_multiomics, AL_dose)."),
    ("supp_table_S4_cold_start.csv",
     "Supp Table S4 — Cold-start numeric summary",
     "9 models × 2 split types (formula-disjoint, ADR-disjoint) × 5 "
     "seeds; reported as seed-mean ± seed-std of AUROC, AUPRC, F1, MCC."),
    ("supp_table_S5_herbpair_vs_herbemb.csv",
     "Supp Table S5 — HerbPairIAM vs HerbEmbIAM per-fold comparison",
     "30-fold paired comparison (3 seeds × 10 folds) of HerbPairIAM "
     "against the learnable-embedding ablation HerbEmbIAM."),
    ("supp_table_S6_dose_head2head_pooled.csv",
     "Supp Table S6 — Dose head-to-head pooled comparison",
     "4 dose / capacity variants (HerbPairIAM = DoseAware_ZeroDose, "
     "DoseAwareIAM, IAM_Wide, InteractionAwareSetModel) pooled across 3 "
     "seeds × 10 folds."),
    ("supp_table_S7_feature_ablation_xgboost.csv",
     "Supp Table S7 — Feature ablation on XGBoost (model-agnostic check)",
     "The same feature-group ablation grid as Main Table 2, re-run on "
     "XGBoost to confirm the feature-importance ordering is not specific "
     "to HerbPairIAM's architecture."),
    ("supp_table_S8_feature_ablation_doseaware.csv",
     "Supp Table S8 — Legacy DoseAwareIAM feature ablation",
     "Archival; the feature ablation table produced against DoseAwareIAM "
     "before HerbPairIAM (zero-dose variant) was adopted as the primary "
     "model. Retained for provenance only; not cited as a primary claim."),
]

_SUPP_CSVS_REFERENCED_BY_FIGURES = [
    ("supp_table_calibration_curves.csv",
     "Reliability-diagram bin points for every model",
     "Consumed by Main Fig 5a and Supp Fig S5."),
    ("supp_table_calibration_summary.csv",
     "Brier score and ECE for every model",
     "Consumed by Main Fig 5a legend and Supp Fig S5a,b."),
    ("supp_table_decision_curve.csv",
     "Net benefit per decision threshold, per model + treat-all / treat-none",
     "Consumed by Main Fig 5b."),
    ("supp_table_adr_cold_start_per_adr.csv",
     "Per-ADR AUROC for every ADR in the cold-start holdout",
     "Consumed by Main Fig 5d and Supp Fig S6."),
    ("supp_table_adr_cold_start_failure_mode.csv",
     "Spearman correlations of per-ADR cold-start AUROC against "
     "KG-coverage and Jaccard variables",
     "Consumed by Main Fig 5d annotation."),
    ("supp_table_leave_one_herb_counterfactual.csv",
     "Per-(fold, sample, herb) leave-one-herb counterfactual table "
     "(29 365 rows)",
     "Consumed by Main Fig 6b."),
    ("supp_table_leave_one_herb_counterfactual_per_sample.csv",
     "Per-sample Spearman summary (3 614 rows)",
     "Consumed by Main Fig 6b caption / Methods."),
    ("supp_table_leave_one_herb_counterfactual_summary.csv",
     "Per-fold aggregate of counterfactual statistics",
     "Consumed by Methods."),
    ("supp_table_case_summary.csv",
     "15 top-confidence OOF cases with romanised formula names",
     "Consumed by Main Fig 6c,d titles and Supp Fig S9."),
    ("supp_table_pmda_concordance.csv",
     "PMDA pharmacovigilance concordance for flagged interactions",
     "Consumed by Methods / Supplementary discussion."),
    ("supp_table_per_adr_analysis.csv",
     "Per-ADR subgroup analysis",
     "Consumed by Supp Fig S6."),
    ("supp_table_per_formula_analysis.csv",
     "Per-formula subgroup analysis",
     "Consumed by Supp Fig S6."),
    ("supp_table_per_group_correlations.csv",
     "Cached Spearman correlations used in subgroup analysis",
     "Consumed by Supp Fig S6 annotations."),
    ("supp_table_nested_cv_xgboost.csv",
     "Nested-CV outer-fold AUROC / AUPRC and selected hyperparameters, "
     "30 folds",
     "Consumed by Supp Fig S7."),
    ("supp_table_neg_sampling_sensitivity.csv",
     "neg_ratio sensitivity across 5 models × 4 ratios with saturation flag",
     "Consumed by Supp Fig S8."),
    ("supp_table_attention_rank_distribution.csv",
     "Attention-weight quantiles per in-sample rank",
     "Consumed by Main Fig 6a caption."),
]

_PROVENANCE_TABLES = [
    ("main_benchmark_canonical_seed42.csv",
     "Canonical seed=42 single-seed main benchmark. Subsumed by "
     "Supp Table S1 (per-seed breakdown) and Main Fig 4 (multi-seed "
     "summary). Kept for provenance."),
    ("main_benchmark_multiseed_underlying_fig4.csv",
     "3-seed summary behind Main Fig 4. Data exposed as a figure, not a "
     "table. Kept for programmatic re-analysis."),
    ("legacy_significance_matrix_auroc.csv",
     "11×11 pairwise paired-Wilcoxon AUROC matrix. Subsumed by Supp "
     "Table S2 (DeLong) and Main Table 3."),
    ("legacy_significance_matrix_auprc.csv",
     "11×11 pairwise paired-Wilcoxon AUPRC matrix. Subsumed by the same."),
    ("legacy_pooled_significance.csv",
     "Pooled-OOF significance summary. Subsumed by Supp Table S2."),
    ("delong_pvalue_matrix.csv",
     "Raw DeLong p-value matrix. The Holm-corrected version is the cited "
     "Supp Table S2."),
    ("bootstrap_cluster_ci_main_benchmark.csv",
     "Cluster-by-fold bootstrap 95 % CIs underlying Main Fig 4 error "
     "bars and the `auroc_ci_low / _high` columns in the figure's "
     "source CSV."),
    ("bootstrap_cluster_ci_structure_ablation.csv",
     "Cluster-by-fold bootstrap CI for the structure ablation table."),
    ("bootstrap_cluster_ci_feature_ablation.csv",
     "Cluster-by-fold bootstrap CI for the feature ablation table."),
    ("bootstrap_cluster_ci_alliance_ablation.csv",
     "Cluster-by-fold bootstrap CI for the alliance ablation table."),
    ("bootstrap_cluster_ci_neg_sampling.csv",
     "Cluster-by-fold bootstrap CI for the negative-sampling sensitivity table."),
    ("bootstrap_cluster_ci_cold_start.csv",
     "Cluster-by-fold bootstrap CI for cold-start."),
    ("herb_attention_consistency.csv",
     "Per-herb × per-ADR attention summary. Subsumed by the leave-one-"
     "herb counterfactual analysis in Main Fig 6b."),
    ("top_novel_predictions_data_availability.csv",
     "Top-K novel predictions (post-acceptance data availability)."),
    ("dose_head2head_per_seed.csv",
     "Per-seed dose head-to-head. Supp Table S6 is the pooled aggregate."),
]


def _write_tables_readme() -> None:
    out = PKG / "TABLES_README.md"
    lines: list[str] = []
    lines.append("# Tables — data dictionary\n")
    lines.append(
        "This document is a data dictionary for every CSV shipped with the "
        "paper. Every cited table has a canonical filename matching its "
        "paper-side label (Main Table N or Supp Table Sn); every "
        "auxiliary CSV referenced by a figure caption is listed under "
        "**Supplementary data artefacts**; every obsolete or subsumed CSV "
        "lives under **Provenance tables**.\n"
    )

    lines.append("\n## Main tables\n")
    for fname, title, desc in _MAIN_TABLES:
        lines.append(f"- `main/tables/{fname}`  \n")
        lines.append(f"  **{title}.** {desc}\n")

    lines.append("\n## Supplementary tables (cited as Supp Table Sn)\n")
    for fname, title, desc in _SUPP_TABLES:
        lines.append(f"- `supplementary/{fname}`  \n")
        lines.append(f"  **{title}.** {desc}\n")

    lines.append("\n## Supplementary data artefacts (cited by figure captions)\n")
    lines.append(
        "These CSVs are the source data for specific figures (main or "
        "supplementary) and are *not* assigned a Table Sn number. They "
        "are referenced by their filename in each figure's caption.\n\n"
    )
    for fname, title, desc in _SUPP_CSVS_REFERENCED_BY_FIGURES:
        lines.append(f"- `supplementary/{fname}`  \n")
        lines.append(f"  **{title}.** {desc}\n")

    lines.append("\n## Provenance tables\n")
    lines.append(
        "Retained under `provenance/tables/` for data availability only; "
        "subsumed by or redundant with a cited table or figure.\n\n"
    )
    for fname, desc in _PROVENANCE_TABLES:
        lines.append(f"- `provenance/tables/{fname}`  \n  {desc}\n")

    lines.append("\n## Column-name conventions\n")
    lines.append(
        "All columns use `snake_case` lowercase. Domain-standard acronyms "
        "(`auroc`, `auprc`, `mcc`, `ece`, `ci`, `se`) remain lowercase in "
        "column keys; the paper's figures and tables render them in the "
        "conventional uppercase (AUROC, AUPRC, ECE, 95 % CI) at the "
        "presentation layer.\n\n"
        "**Metric precision:** four decimals for AUROC, AUPRC, F1, MCC, "
        "Brier, ECE, Δ-to-primary columns, and CI bounds.\n\n"
        "**p-value precision:** three significant figures in scientific "
        "notation.\n\n"
        "**Spearman ρ precision:** three decimals.\n\n"
        "Integer-valued columns (n_folds, n_seeds, wins) are stored as "
        "integers.\n"
    )

    lines.append("\n## Regenerating this package\n")
    lines.append("```bash\n")
    lines.append("python src/scripts/assemble_paper_package.py\n")
    lines.append("python src/scripts/polish_tables.py\n")
    lines.append("```\n\nThe polish script is idempotent.\n")

    out.write_text("".join(lines))
    print(f"[polish] wrote data dictionary → {out.relative_to(PKG)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    # (1) File reorganisation first.
    _apply_renames()

    # (2–4) In-place CSV polishing.
    all_csvs = list((PKG / "main" / "tables").glob("*.csv"))
    all_csvs += list((PKG / "supplementary").glob("supp_table_*.csv"))
    all_csvs += list((PKG / "provenance" / "tables").glob("*.csv"))
    for p in sorted(all_csvs):
        _polish_one(p)

    # Data dictionary.
    _write_tables_readme()
    print(f"[polish] done. {len(all_csvs)} CSVs polished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
