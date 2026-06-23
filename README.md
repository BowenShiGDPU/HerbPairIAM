# HerbPairIAM

Interpretable-by-construction, interaction-aware attention over a pharmacovigilance
chemical-safety graph for Kampo formula–adverse drug reaction (ADR) signal
prioritization, with per-component and per-component-pair attribution.

This repository contains the preprocessing, feature construction, training,
evaluation, package-insert auditing, interpretability, structural follow-up, and
figure-generation code for the accompanying paper. Large processed artifacts
(node/relation tables, feature tensors, fold predictions, trained model files,
audit outputs, reproducibility metadata) are archived on Zenodo:

> **Data & artifact package** — https://doi.org/10.5281/zenodo.19326538

## Repository layout

```
src/
├── models/        neural set models + training loop (neural_models.py),
│                  primary model wrapper (herb_pair_iam.py), graph and tabular baselines
├── data/          phase1_precompute.py (KG-derived profiles), phase2_dataset.py (dataset + CV splits)
├── evaluation/    phase4_evaluation.py (benchmark, calibration, decision-curve, ablations),
│                  phase5_interpretability.py (attention + leave-one-component perturbation),
│                  finalize_results.py, bootstrap_pooled_ci.py, make_figures.py, experiment_utils.py
└── scripts/       run_primary_canonical.py, run_multiseed_baselines.py,
                   package-insert audit (audit_top500_nonsignal.py, scan_topk_pmda_curve.py,
                   repeated_pmda_random_audit.py, audit_random_nonsignal_baseline.py),
                   structural follow-up (run_formal_structure_ablation_job.py,
                   prepare_mechanistic_publication_outputs.py), subgroup/statistics
                   (compute_subgroup_analysis.py, compute_pairwise_delong.py), and aggregation drivers
```

All paths resolve relative to the repository root; no machine-specific paths are
hardcoded.

## Prepare data

1. Download the **data package** from the Zenodo DOI above.
2. Place the unpacked processed inputs at the repository root as `final_data_clean/`
   (node/relation CSVs, signal tables) and let preprocessing write to `outputs/`.
3. Access to the original JADER, FAERS, MedDRA, and Japanese PMDA package-insert
   resources is governed by the terms of the respective data providers and is not
   redistributed here; place locally obtained PMDA inserts under `药品说明书/` to run
   the package-insert audit.

## Reproduce

```bash
python -u src/data/phase1_precompute.py        # build KG-derived component/ADR/pair profiles -> outputs/
python -u src/data/phase2_dataset.py           # build dataset.pkl + leakage-controlled CV splits
RESULTS_ROOT_DIR=results EXPERIMENT_SUBDIR=main_benchmark VAL_SELECTION_METRIC=auroc \
  python -u src/scripts/run_primary_canonical.py            # primary model, 10-fold, frozen config
RESULTS_ROOT_DIR=results EXPERIMENT_SUBDIR=multiseed_baselines \
  python -u src/scripts/run_multiseed_baselines.py \
    --seeds 42 13 7 --models XGBoost RandomForest GradientBoosting LogisticRegression MLP R-GCN HGT --neg-ratio 10
python -u src/evaluation/phase4_evaluation.py  # benchmark table, calibration, decision-curve, ablations
python -u src/evaluation/phase5_interpretability.py        # attention + leave-one-component perturbation
python -u src/scripts/audit_top500_nonsignal.py            # package-insert audit of top-ranked candidates
python -u src/evaluation/make_figures.py       # manuscript figures
```

The primary model is `HerbPairIAM`. Public comparators are seven formal baselines:
logistic regression, random forest, gradient boosting, XGBoost, MLP, R-GCN, and HGT.
See `NAMING.md` for how these paper-facing names map to the source-code class names.

## Model configuration

See `model_configurations.csv` for the primary model and baseline settings, including
input feature dimensions and dose handling (the primary model zeroes both the
formula-level auxiliary dose vector and the pair-level dose-derived channels).

## Reproducibility metadata

* Every training run writes a `run_manifest.json` recording the dataset SHA-256, the
  selected configuration, package versions, and host info; `run_manifest_primary.json`
  in this repository records the canonical primary run (see also `run_manifest_history`
  in the archive, which preserves the provenance of each result subdirectory).
* `src/evaluation/experiment_utils.py::set_seed` seeds Python, NumPy, and PyTorch and
  sets deterministic cuDNN; opt-in strict determinism reproduces the reported numbers.
* The package is validated from a clean checkout: all scripts byte-compile and a reduced-epoch run of the primary model on canonical fold 0 trains end-to-end using only repository-root-relative paths (see `SMOKE_TEST_RESULT.md`).

## Environment

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, scikit-learn, torch, xgboost, ...
```

## Citation

If you use this code or the data package, please cite the accompanying paper
(BibTeX to follow upon acceptance).

## License

MIT. See `LICENSE`.
