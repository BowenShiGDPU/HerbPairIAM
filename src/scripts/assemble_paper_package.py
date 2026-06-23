"""Assemble the paper-ready data package under ``paper_package/``.

Collects every CSV, figure, and artefact needed to populate the main text
and the supplementary materials of the Nature Communications submission.
Files are organised into three top-level groups:

    paper_package/
        main/                   — data that lands in the main text
            figures/            — main-text figures (PNG / PDF)
            tables/             — main-text tables (CSV)
        supplementary/          — supporting evidence (CSV, JSON)
        provenance/             — run_manifest history + env.yml
        README.md               — table-of-contents mapping artefact -> section

The script is idempotent: it copies from the canonical
``results/formal_doseaware_neg10_auroc/`` subdirectories and writes into
``paper_package/``. Running it again refreshes the package in place.

Nothing in here is a computation — the underlying results must have been
produced by the training / aggregation scripts already.

Usage::

    python -u src/scripts/assemble_paper_package.py

Optional flags::

    --results-subdir    (default formal_doseaware_neg10_auroc)
    --output-dir        (default paper_package)
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable


sys.stdout.reconfigure(line_buffering=True)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _copy_if_exists(src: Path, dst: Path, note: str = "") -> bool:
    """Copy ``src`` to ``dst`` if the source exists. Returns success."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    if note:
        print(f"  [copy] {note}: {src.relative_to(ROOT_DIR)} -> {dst.relative_to(ROOT_DIR)}")
    return True


def _write_readme(out_dir: Path, entries: list[tuple[str, str, str]]) -> None:
    """Emit the table of contents README."""
    lines = [
        "# TCMFADR — HerbPairIAM Publication Data Package\n",
        "This directory is the paper-ready data bundle for the Nature ",
        "Communications submission. Every file below is referenced by the ",
        "main text or the supplementary materials.\n",
        "\n",
        "## Primary model\n",
        "**HerbPairIAM** — see `/PRIMARY_MODEL.md` at the repository root and ",
        "`src/models/herb_pair_iam.py` for the architecture definition and ",
        "the experimental evidence that established this choice.\n",
        "\n",
        "## Headline numbers\n",
        "HerbPairIAM (3 outer-CV seeds × 10 folds = 30 paired folds, ",
        "cluster bootstrap 95% CI):\n",
        "- AUROC 0.822 [0.813, 0.827]\n",
        "- AUPRC 0.517 [0.474, 0.514]\n\n",
        "\n",
        "## Contents\n",
        "\n",
        "| Section | Artefact | Path |\n",
        "|---|---|---|\n",
    ]
    for section, artefact, rel_path in entries:
        lines.append(f"| {section} | {artefact} | `{rel_path}` |\n")
    lines.extend([
        "\n",
        "## Reproducibility\n",
        "\n",
        "- `provenance/` contains the `run_manifest.json` history for every ",
        "stage, capturing git commit, dataset SHA-256, environment, package ",
        "versions, and CUDA / GPU information at the time of each training ",
        "and aggregation run.\n",
        "- `provenance/requirements.txt` and `provenance/environment.yml` ",
        "pin the full software stack. Strict determinism (bit-exact seed ",
        "reproduction) can be enabled with `STRICT_DETERMINISM=1`.\n",
        "- Fold pickles and model state dicts live under the original ",
        "results tree (not copied here to keep this package small); each ",
        "table in `tables/` names the fold pickles it aggregates.\n",
    ])
    with open(out_dir / "README.md", "w") as f:
        f.write("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-subdir", default="formal_doseaware_neg10_auroc")
    parser.add_argument("--output-dir", default="paper_package")
    args = parser.parse_args()

    results_root = ROOT_DIR / "results" / args.results_subdir
    out_root = ROOT_DIR / args.output_dir

    # Preserve hand-authored top-level docs across repackagings. Rationale:
    # the README and the experiment logbook are the single most important
    # files for an AI / human paper-writer consuming this package, so we
    # protect them from silent overwrites. Sub-directory trees are still
    # refreshed.
    preserved_docs: dict[str, str] = {}
    for doc_name in ("README.md", "EXPERIMENT_LOGBOOK.md"):
        doc_path = out_root / doc_name
        if doc_path.exists() and doc_path.stat().st_size > 5000:
            preserved_docs[doc_name] = doc_path.read_text()
            print(f"[paper_package] Preserving {doc_name} ({doc_path.stat().st_size} bytes)")

    if out_root.exists():
        print(f"Removing existing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for doc_name, content in preserved_docs.items():
        (out_root / doc_name).write_text(content)

    main_tables = out_root / "main" / "tables"
    main_figs = out_root / "main" / "figures"
    supp_root = out_root / "supplementary"
    prov_root = out_root / "provenance"

    # Every entry below maps to a (section label, artefact description,
    # destination path). Missing sources are skipped so the package builds
    # incrementally while experiments are still filling in.
    entries: list[tuple[str, str, str]] = []

    # --- Main text Table 1: the 30-fold multi-seed benchmark -------------
    src = results_root / "main_benchmark" / "tables" / "main_benchmark_multiseed.csv"
    dst = main_tables / "table1_main_benchmark_multiseed.csv"
    if _copy_if_exists(src, dst, "Main Table 1"):
        entries.append(("Main Table 1", "Multi-seed main benchmark (3 seeds × 10 folds, 95% CI)",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Table 1 (single-seed canonical) ----------------------
    src = results_root / "main_benchmark" / "tables" / "main_benchmark.csv"
    dst = main_tables / "table1_main_benchmark_canonical.csv"
    if _copy_if_exists(src, dst, "Main Table 1 (canonical)"):
        entries.append(("Main Table 1 (alt)", "Canonical seed=42 main benchmark with pHolm",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Table 2: structure ablation ---------------------------
    src = results_root / "structure_ablation" / "tables" / "structure_ablation.csv"
    dst = main_tables / "table2_structure_ablation.csv"
    if _copy_if_exists(src, dst, "Main Table 2"):
        entries.append(("Main Table 2", "Structure ablation with paired Wilcoxon + Holm",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Table 3: feature ablation ----------------------------
    src = results_root / "feature_ablation" / "tables" / "feature_ablation.csv"
    dst = main_tables / "table3_feature_ablation.csv"
    if _copy_if_exists(src, dst, "Main Table 3"):
        entries.append(("Main Table 3", "Feature ablation (KG features) with pair_zero diagnostic",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Table 4: alliance ablation ---------------------------
    src = results_root / "alliance_ablation" / "tables" / "alliance_ablation.csv"
    dst = main_tables / "table4_alliance_ablation.csv"
    if _copy_if_exists(src, dst, "Main Table 4"):
        entries.append(("Main Table 4", "Alliance-style leave-one-out ablation",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Table 5: cold-start ----------------------------------
    src = results_root / "cold_start" / "tables" / "cold_start.csv"
    dst = main_tables / "table5_cold_start.csv"
    if _copy_if_exists(src, dst, "Main Table 5"):
        entries.append(("Main Table 5", "Formula and ADR cold-start evaluation",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Table 6: HerbPairIAM vs HerbEmbIAM (learnable embedding) -
    src = results_root / "dose_head2head" / "tables" / "herbpair_vs_herbemb_paired.csv"
    dst = main_tables / "table6_herbpair_vs_herbemb_paired.csv"
    if _copy_if_exists(src, dst, "Main Table 6"):
        entries.append(("Main Table 6", "HerbPairIAM vs HerbEmbIAM (learnable embedding), 30 paired folds",
                        dst.relative_to(out_root).as_posix()))

    # --- Main text Figures: quick-look calibration + decision-curve PNGs.
    # These are diagnostic previews; the composite main-text figures (Fig 4,
    # Fig 5, Fig 6) are assembled separately by paper_figures/fig*/plot_*.py
    # and copied in below.
    for name in ("calibration.png", "decision_curve.png"):
        src = results_root / "main_benchmark" / "figures" / name
        dst = main_figs / name
        if _copy_if_exists(src, dst, f"Main figure preview: {name}"):
            entries.append(("Main figure preview", name, dst.relative_to(out_root).as_posix()))

    # --- Composite main-text figures produced by paper_figures/fig*/ ------
    figures_root = ROOT_DIR / "paper_figures"
    for fig_id, fig_dir in [
        ("fig4", figures_root / "fig4_main_benchmark" / "out"),
        ("fig5", figures_root / "fig5_calib_cold" / "out"),
        ("fig6", figures_root / "fig6_interpret" / "out"),
    ]:
        for ext in ("png", "pdf", "svg"):
            src = fig_dir / f"{fig_id}.{ext}"
            dst = main_figs / f"{fig_id}.{ext}"
            if _copy_if_exists(src, dst, f"Main figure: {fig_id}.{ext}"):
                if ext == "png":
                    entries.append(("Main figure", f"{fig_id}.png",
                                    dst.relative_to(out_root).as_posix()))

    # --- Supplementary figures (script-drawn S4-S9, reused S1-S3) ---------
    supp_fig_dir = out_root / "supplementary" / "figures"
    supp_fig_dir.mkdir(parents=True, exist_ok=True)
    script_supp_map = [
        ("figS4", figures_root / "figS4_full_curves" / "out"),
        ("figS5", figures_root / "figS5_calibration_detail" / "out"),
        ("figS6", figures_root / "figS6_subgroup" / "out"),
        ("figS7", figures_root / "figS7_nested_cv" / "out"),
        ("figS8", figures_root / "figS8_neg_sensitivity" / "out"),
        ("figS9", figures_root / "figS9_case_grid" / "out"),
    ]
    for fig_id, fig_dir in script_supp_map:
        for ext in ("png", "pdf", "svg"):
            src = fig_dir / f"{fig_id}.{ext}"
            dst = supp_fig_dir / f"{fig_id}.{ext}"
            if _copy_if_exists(src, dst, f"Supp figure: {fig_id}.{ext}"):
                if ext == "png":
                    entries.append(("Supp figure", f"{fig_id}.png",
                                    dst.relative_to(out_root).as_posix()))

    # Reused supp figures from the old paper (pure data description).
    old_figs = ROOT_DIR / "old_paper" / "elsarticle" / "figures"
    for fig_id in ("figS1", "figS2", "figS3"):
        src = old_figs / f"{fig_id}.pdf"
        dst = supp_fig_dir / f"{fig_id}.pdf"
        if _copy_if_exists(src, dst, f"Supp figure (reused): {fig_id}.pdf"):
            entries.append(("Supp figure (reused)", f"{fig_id}.pdf",
                            dst.relative_to(out_root).as_posix()))

    # --- Supplementary: full per-experiment CSVs ------------------------
    supp_map = {
        ("main_benchmark", "tables/main_benchmark_per_seed.csv"): "supp_table_main_benchmark_per_seed.csv",
        ("main_benchmark", "tables/significance_matrix_auroc.csv"): "supp_table_significance_matrix_auroc.csv",
        ("main_benchmark", "tables/significance_matrix_auprc.csv"): "supp_table_significance_matrix_auprc.csv",
        ("main_benchmark", "supplementary/pooled_bootstrap_ci.csv"): "supp_table_bootstrap_cluster_ci.csv",
        ("main_benchmark", "supplementary/delong_holm_adjusted.csv"): "supp_table_delong_holm.csv",
        ("main_benchmark", "supplementary/delong_pvalue_matrix.csv"): "supp_table_delong_matrix.csv",
        ("main_benchmark", "supplementary/pooled_significance.csv"): "supp_table_pooled_significance.csv",
        ("main_benchmark", "supplementary/calibration_curves.csv"): "supp_table_calibration_curves.csv",
        ("main_benchmark", "supplementary/calibration_summary.csv"): "supp_table_calibration_summary.csv",
        ("main_benchmark", "supplementary/decision_curve.csv"): "supp_table_decision_curve.csv",
        ("main_benchmark", "supplementary/per_adr_analysis.csv"): "supp_table_per_adr_analysis.csv",
        ("main_benchmark", "supplementary/per_formula_analysis.csv"): "supp_table_per_formula_analysis.csv",
        ("main_benchmark", "supplementary/per_group_correlations.csv"): "supp_table_per_group_correlations.csv",
        ("main_benchmark", "supplementary/nested_cv_xgboost.csv"): "supp_table_nested_cv_xgboost.csv",
        ("main_benchmark", "supplementary/primary_vs_baselines_paired_30fold.csv"): "supp_table_primary_vs_baselines_paired_30fold.csv",
        ("main_benchmark", "supplementary/leave_one_herb_counterfactual.csv"): "supp_table_leave_one_herb_counterfactual.csv",
        ("main_benchmark", "supplementary/leave_one_herb_counterfactual_per_sample.csv"): "supp_table_leave_one_herb_counterfactual_per_sample.csv",
        ("main_benchmark", "supplementary/leave_one_herb_counterfactual_summary.csv"): "supp_table_leave_one_herb_counterfactual_summary.csv",
        ("cold_start", "supplementary/adr_cold_start_per_adr.csv"): "supp_table_adr_cold_start_per_adr.csv",
        ("cold_start", "supplementary/adr_cold_start_failure_mode.csv"): "supp_table_adr_cold_start_failure_mode.csv",
        ("cold_start", "supplementary/pooled_bootstrap_ci.csv"): "supp_table_cold_start_bootstrap_ci.csv",
        ("structure_ablation", "supplementary/pooled_bootstrap_ci.csv"): "supp_table_structure_ablation_bootstrap_ci.csv",
        ("feature_ablation", "supplementary/pooled_bootstrap_ci.csv"): "supp_table_feature_ablation_bootstrap_ci.csv",
        ("feature_ablation", "tables/feature_ablation_xgboost_supplementary.csv"): "supp_table_feature_ablation_xgboost.csv",
        ("feature_ablation", "tables/feature_ablation_doseaware_supplementary.csv"): "supp_table_feature_ablation_doseaware.csv",
        ("alliance_ablation", "supplementary/pooled_bootstrap_ci.csv"): "supp_table_alliance_ablation_bootstrap_ci.csv",
        ("neg_sampling_sensitivity", "supplementary/neg_sampling_sensitivity.csv"): "supp_table_neg_sampling_sensitivity.csv",
        ("neg_sampling_sensitivity", "supplementary/pooled_bootstrap_ci.csv"): "supp_table_neg_sampling_bootstrap_ci.csv",
        ("main_benchmark", "interpretability/case_summary.csv"): "supp_table_case_summary.csv",
        ("main_benchmark", "interpretability/herb_attention_consistency.csv"): "supp_table_herb_attention_consistency.csv",
        ("main_benchmark", "supplementary/attention_rank_distribution.csv"): "supp_table_attention_rank_distribution.csv",
        ("main_benchmark", "interpretability/pmda_concordance.csv"): "supp_table_pmda_concordance.csv",
        ("main_benchmark", "interpretability/top_novel_predictions.csv"): "supp_table_top_novel_predictions.csv",
        ("dose_head2head", "tables/dose_head2head_summary.csv"): "supp_table_dose_head2head_summary.csv",
        ("dose_head2head", "tables/dose_head2head_pooled.csv"): "supp_table_dose_head2head_pooled.csv",
    }
    for (subdir, rel), dst_name in supp_map.items():
        src = results_root / subdir / rel
        dst = supp_root / dst_name
        if _copy_if_exists(src, dst, f"Supp: {dst_name}"):
            entries.append(("Supp", dst_name.replace(".csv", "").replace("supp_table_", ""),
                            dst.relative_to(out_root).as_posix()))

    # --- Supplementary: case JSON files ---------------------------------
    case_src = results_root / "main_benchmark" / "interpretability"
    case_dst = supp_root / "case_studies"
    case_dst.mkdir(parents=True, exist_ok=True)
    if case_src.exists():
        n_copied = 0
        for p in sorted(case_src.glob("case_*.json")):
            _copy_if_exists(p, case_dst / p.name)
            n_copied += 1
        if n_copied:
            entries.append(("Supp", f"{n_copied} interpretability case studies (JSON)",
                            (case_dst.relative_to(out_root)).as_posix()))

    # --- Provenance -------------------------------------------------------
    for subdir in ["main_benchmark", "structure_ablation", "feature_ablation",
                   "alliance_ablation", "neg_sampling_sensitivity", "cold_start",
                   "dose_head2head", "multiseed_baselines"]:
        for name in ["run_manifest.json", "run_manifest_history.jsonl"]:
            src = results_root / subdir / name
            if src.exists():
                dst = prov_root / subdir / name
                _copy_if_exists(src, dst)

    # Copy env / reqs / plan.
    for src_name in ["requirements.txt", "environment.yml", "PUBLICATION_PLAN.md", "PRIMARY_MODEL.md"]:
        src = ROOT_DIR / src_name
        dst = prov_root / src_name
        if _copy_if_exists(src, dst):
            entries.append(("Provenance", src_name, dst.relative_to(out_root).as_posix()))

    # --- Write the README -------------------------------------------------
    # The detailed AI-friendly README is hand-authored; only regenerate the
    # auto README if none exists yet. This prevents a silent overwrite of the
    # curated guidance document.
    readme_path = out_root / "README.md"
    if readme_path.exists() and readme_path.stat().st_size > 5000:
        print(f"[paper_package] Preserving existing README.md ({readme_path.stat().st_size} bytes)")
    else:
        _write_readme(out_root, entries)
        print(f"[paper_package] Wrote fresh auto README.md")

    # Summary stdout.
    print(f"\n[paper_package] Wrote {out_root}")
    print(
        "[paper_package] Next step: run `python src/scripts/polish_tables.py`\n"
        "[paper_package]   to apply the Main/Supp/Provenance reorganisation,\n"
        "[paper_package]   normalise column names and precision, and write\n"
        "[paper_package]   TABLES_README.md."
    )
    from collections import Counter
    file_counts = Counter()
    for p in out_root.rglob("*"):
        if p.is_file():
            file_counts[p.parent.relative_to(out_root).as_posix()] += 1
    for d, n in sorted(file_counts.items()):
        print(f"  {d:40s}: {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
