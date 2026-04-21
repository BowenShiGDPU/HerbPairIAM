# HerbPairIAM

Interpretable-by-construction interaction-aware attention over a
pharmacovigilance knowledge graph for Kampo formula–adverse drug reaction
(ADR) signal prediction, with per-herb and per-pair attribution.

This repository contains the model code, data-preparation pipeline, and
training entry points used in the accompanying paper. Processed CSV
resources (node tables, relation tables, signal definitions, split files)
are released separately on Zenodo:

> **Data package** – <https://doi.org/10.5281/zenodo.19326538>

## Repository layout

```
src/
├── models/
│   ├── herb_pair_iam.py          primary model (thin wrapper over DoseAwareIAM)
│   ├── herb_emb_iam.py           learnable-embedding ablation
│   ├── neural_models.py          interaction-aware set models + training loop
│   ├── graph_baselines.py        R-GCN and HGT baselines
│   └── tabular_models.py         LogReg / RF / GB / MLP / XGBoost
├── data/
│   ├── phase1_precompute.py      KG-driven herb-pair profile precomputation
│   └── phase2_dataset.py         feature table + 10-fold pair-stratified CV splits
├── evaluation/
│   └── experiment_utils.py       metrics, DeLong, Holm–Bonferroni, bootstrap CI,
│                                  run-manifest writer, I/O paths
└── scripts/
    ├── prepare_inputs.py         load dataset + KG artefacts, emit run manifest
    ├── run_primary_canonical.py  train HerbPairIAM (10-fold, frozen config)
    ├── run_multiseed_baselines.py tabular + graph baselines under extra seeds
    └── run_herb_emb_iam.py       HerbEmbIAM ablation under multiple seeds
```

Backward-compatible `src/*.py` shims re-export the modules under their
short names so the scripts can use `from experiment_utils import ...`
and `from neural_models import ...`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate herbpairiam
```

A CUDA GPU is recommended for graph baselines (R-GCN, HGT); CPU is fine
for the tabular baselines and HerbPairIAM itself.

## Prepare the data

1. Download the **data package** from the Zenodo DOI above and place the
   unpacked directory at `final_data_clean/` (sibling of `src/`).
2. Build the KG-derived profiles and dataset artefacts (writes to
   `outputs/`):

```bash
python -u src/data/phase1_precompute.py
python -u src/data/phase2_dataset.py
```

The resulting `outputs/dataset.pkl`, `outputs/herb_target_profiles.pkl`,
`outputs/adr_target_profiles.pkl`, `outputs/herb_pair_features.pkl`, and
`outputs/lookups.pkl` are the inputs consumed by every training script.

## Train HerbPairIAM

```bash
RESULTS_ROOT_DIR=results \
EXPERIMENT_SUBDIR=main_benchmark \
VAL_SELECTION_METRIC=auroc \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -u src/scripts/run_primary_canonical.py
```

Each fold writes a pickle with out-of-fold predictions, metrics, and
attention traces to
`results/main_benchmark/fold_results/HerbPairIAM_fold{K}.pkl` and a state
dict to `results/main_benchmark/models/HerbPairIAM_fold{K}.pt`. Existing
fold pickles are skipped; pass `--no-skip` to retrain.

## Baselines

Tabular and graph baselines under multiple outer-CV seeds:

```bash
RESULTS_ROOT_DIR=results \
EXPERIMENT_SUBDIR=multiseed_baselines \
python -u src/scripts/run_multiseed_baselines.py \
    --seeds 42 13 7 \
    --models XGBoost RandomForest GradientBoosting LogisticRegression MLP R-GCN HGT \
    --neg-ratio 10
```

## HerbEmbIAM ablation

Replace the KG-derived individual herb and ADR profiles with learnable
`nn.Embedding` tables (48-dim) while keeping the pair branch unchanged:

```bash
RESULTS_ROOT_DIR=results \
EXPERIMENT_SUBDIR=herb_emb_iam \
python -u src/scripts/run_herb_emb_iam.py --seeds 42 13 7 --neg-ratio 10
```

## Reproducibility

* Every training run writes a `run_manifest.json` with the git commit,
  `outputs/dataset.pkl` SHA-256, environment variables, Python and
  package versions, and host/CUDA info.
* `src/evaluation/experiment_utils.py::set_seed` seeds Python, NumPy,
  and PyTorch (CPU + CUDA) and sets `cudnn.deterministic=True`.
* Opt-in strict determinism (`STRICT_DETERMINISM=1`) turns on
  `torch.use_deterministic_algorithms(True, warn_only=True)` and sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; it reproduces the reported numbers
  bit-exactly at roughly twice the training cost.

## Citation

If you use this code or the accompanying data package, please cite the
paper (BibTeX entry to follow upon acceptance).

## License

MIT. See `LICENSE`.
