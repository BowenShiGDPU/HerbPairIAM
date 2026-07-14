# Code, data, and artifact release map

Public code is in this repository (GitHub). Large processed artifacts are archived on
Zenodo (https://doi.org/10.5281/zenodo.20821998). Raw provider data is not redistributed. Model naming (paper-facing names vs. source-code class names) is documented in `NAMING.md`.

| Artifact | Public location | Role | Canonical / sensitivity | Reproduces |
|---|---|---|---|---|
| `src/data/phase1_precompute.py`, `phase2_dataset.py` | GitHub | preprocessing + feature construction | canonical | profiles, `dataset.pkl`, CV splits |
| `src/scripts/run_primary_canonical.py`, `src/models/*` | GitHub | primary model training | canonical | main benchmark fold results |
| `src/scripts/run_multiseed_baselines.py` | GitHub | seven formal baselines | canonical | baseline fold results |
| `src/evaluation/phase4_evaluation.py`, `finalize_results.py`, `bootstrap_pooled_ci.py` | GitHub | benchmark, calibration, decision-curve, ablations, statistics | canonical | benchmark + calibration + DCA + significance tables |
| `src/evaluation/phase5_interpretability.py`, `src/scripts/compute_leave_one_herb_counterfactual.py` | GitHub | attention + leave-one-component perturbation | canonical | interpretability outputs |
| `src/scripts/audit_top500_nonsignal.py`, `scan_topk_pmda_curve.py`, `audit_random_nonsignal_baseline.py` | GitHub | package-insert audit (incl. ADR-family-matched baselines) | canonical | top-K and family-matched enrichment results |
| `src/evaluation/make_figures.py`, `src/scripts/polish_tables.py` | GitHub | selected benchmark/evaluation figure and table generation | canonical | selected benchmark/evaluation figures and tables |
| `outputs/dataset.pkl`, profiles, pair features | Zenodo | processed model inputs | canonical | feature tensors |
| trained model checkpoints (`*.pt`) | Zenodo | model weights | canonical | reload for inference/interpretability |
| fold out-of-fold predictions (`*.pkl`) | Zenodo | OOF scores + attention traces | canonical | benchmark/interpretability inputs |
| run manifests / `run_manifest_primary.json` / run-manifest history | GitHub + Zenodo | provenance | canonical | exact run reconstruction |
| package-insert audit outputs | Zenodo | result artifacts | canonical | audit tables |
| raw JADER / FAERS / MedDRA / PMDA package inserts | not redistributed | source data | — | governed by provider terms |

The repository is validated from a clean checkout (`SMOKE_TEST_RESULT.md`): all scripts byte-compile and the primary model trains end-to-end on canonical fold 0 using repository-root-relative paths only.
