"""Evaluation, statistics, interpretability, and figure rendering.

Public submodules:

* ``experiment_utils``           -- Output paths, metric helpers, DeLong, bootstrap CI.
* ``phase4_evaluation``          -- 10-fold CV runners + main / feature / structure / cold-start / neg-sampling stages.
* ``phase5_interpretability``    -- Case JSONs, PMDA concordance, attention consistency, novel top-K.
* ``finalize_results``           -- Pooled significance + main_benchmark.csv aggregation.
* ``bootstrap_pooled_ci``        -- Pooled OOF AUROC/AUPRC bootstrap CI (1000x).
* ``compute_profile_postprocess`` -- n_parameters / inference per-sample / hardware fields for compute table.
* ``supplementary_analyses``     -- s9.2 cross-database, s9.4 ADR coverage, s9.5 formula size, s9.6 feature importance.
* ``make_figures``               -- ROC / PR / calibration / cold-start / sensitivity / attention heatmap PDFs.
"""
