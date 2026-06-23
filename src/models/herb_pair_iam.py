"""HerbPairIAM — the primary model of this project.

Status
------
**This is the primary model.** All headline numbers in the paper (main
benchmark, ablation tables, cold-start, interpretability case studies) are
reported on HerbPairIAM unless explicitly stated otherwise.

History / why this name
-----------------------
The project originally used a model named ``DoseAwareIAM`` (Dose-aware
Interaction-aware Set Model) that concatenated real formula / herb / pair
dose signals into both the set-readout branches and the final predictor.
A 3-seed × 10-fold multi-seed head-to-head experiment
(``src/scripts/run_dose_head2head.py``, results under
``results/formal_doseaware_neg10_auroc/dose_head2head/``) compared four
variants on the same 30 paired folds:

    V0  InteractionAwareSetModel  — no dose anywhere (hidden=32, ~10k params)
    V0w IAM_Wide                  — V0 with hidden=44 (~17k params)
    V4z DoseAware_ZeroDose        — full DoseAware architecture, dose inputs
                                     zero-filled   << THIS IS HerbPairIAM >>
    V4  DoseAwareIAM              — full DoseAware architecture, real dose
                                     inputs

Final 30-fold results (paired Wilcoxon, Holm-corrected across 6 pairs):

    V4z (HerbPairIAM)   0.8228 ± 0.003 AUROC   0.5166 ± 0.011 AUPRC   <-- winner
    V0w IAM_Wide        0.8169 ± 0.003 AUROC   0.4980 ± 0.016 AUPRC
    V0  IAM             0.8134 ± 0.003 AUROC   0.4935 ± 0.010 AUPRC
    V4  DoseAwareIAM    0.8114 ± 0.002 AUROC   0.4978 ± 0.005 AUPRC

    V4z > V4 (DoseAwareIAM with real dose):  p_Holm AUROC=0.003,  AUPRC=0.025
    V4z > V0 (IAM):                          p_Holm AUROC=0.005,  AUPRC=0.023
    V4z > V0w (capacity-matched IAM):        p_Holm AUROC=0.28,   AUPRC=0.018

Interpretation:

1. The full DoseAware structure (dose_enc MLP + dose_z concatenated into the
   prediction head) is genuinely helpful, improving AUPRC over a plain IAM
   by about +0.023 (p_Holm=0.023) and over a capacity-matched wider IAM by
   +0.019 (p_Holm=0.018). So the gain is structural, not a capacity effect.
2. **Feeding real dose values on top of that structure *degrades*
   performance** by about 0.012 AUROC / 0.019 AUPRC (p_Holm=0.003 / 0.025).
   On the current 707-labelled dataset, dose magnitudes act as label-agnostic
   noise that distracts the predictor.
3. HerbPairIAM therefore keeps the full architecture (including the
   ``dose_enc`` MLP and the ``dose_z`` slot in ``pred``) but feeds a
   zero-filled "dose" input. Those extra hidden dimensions end up acting
   as a *learned constant bias channel* that stabilises training.

The name reflects what the model *actually* does: it reasons about herbs
and herb-pair interactions. It deliberately does **not** claim to use
dose information.

Architecture (equivalent to ``DoseAwareInteractionAwareSetModel`` with
``adr_conditioned=True``, ``use_dose_gate=True``, ``pool_type="attention"``,
fed with samples built under ``feature_ablation={"AL_dose"}``):

    Inputs per sample:
      node_features  (n_herbs, 54)   # last 6 dims are the zeroed dose tail
      pair_features  (n_pairs, 15)   # last 4 dims are the zeroed dose extras
      adr_features   (48,)
      formula_dose_features  (6,)    # zero-filled vector

    Branches:
      node_h  = MLP(node_features)
      pair_h  = MLP(pair_features)
      a       = MLP(adr_features)
      dose_z  = MLP(formula_dose_features)    # constant (trainable bias)

      node_alpha = softmax(Linear([node_h; a]))
      pair_alpha = softmax(Linear([pair_h; a]))
      z_node = sum(node_alpha * node_h)  * sigmoid(Linear([a; dose_z]))
      z_pair = sum(pair_alpha * pair_h)  * sigmoid(Linear([a; dose_z]))

      logit  = Linear([z_node; z_pair; dose_z; a])

Frozen training hyperparameters (see
``scripts/run_formal_stage.FROZEN_DOSEAWARE_CONFIG`` and
``phase4_evaluation.default_best_configs``):

    hidden=32  dropout=0.3  lr=1e-3  max_epochs=100  patience=10
    batch_size=32  neg_ratio=10  val_selection_metric=AUROC

Canonical result files
----------------------
* 30-fold head-to-head results:
  ``results/formal_doseaware_neg10_auroc/dose_head2head/fold_results/H2H_seed{S}_HerbPairIAM_fold{F}.pkl``
* Main benchmark (rerun tracked in follow-up):
  ``results/formal_doseaware_neg10_auroc/main_benchmark/fold_results/HerbPairIAM_fold{F}.pkl``

API
---
The model *does not have its own Python class* — it is implemented as an
alias on top of ``DoseAwareInteractionAwareSetModel`` so that existing
fold pickles (trained under the interim name ``DoseAware_ZeroDose``) remain
bit-compatible. To build the model use::

    from models.neural_models import build_model, ModelConfig
    m = build_model("HerbPairIAM", ModelConfig(), sample_example)

and to obtain zero-dose samples::

    from models.neural_models import build_sample_collections
    sample_map = build_sample_collections(df, lookups, hp, ap, pf,
                                          ["HerbPairIAM"])
    # sample_map["HerbPairIAM"] already has dose fields zero-filled because
    # model_intrinsic_ablation("HerbPairIAM") == frozenset({"AL_dose"}).
"""

from __future__ import annotations

MODEL_NAME = "HerbPairIAM"
ALIASES = ("DoseAware_ZeroDose",)  # historical alias used before naming was settled
INTRINSIC_ABLATION_TAGS = frozenset({"AL_dose"})

# Frozen training configuration as established by the dose_head2head experiment.
FROZEN_CONFIG = {
    "hidden": 32,
    "dropout": 0.3,
    "lr": 1e-3,
    "epochs": 100,
    "patience": 10,
    "batch_size": 32,
    "eval_every": 2,
    "neg_ratio": 10,
    "val_selection_metric": "auroc",
}


def build(config=None, sample_example=None):
    """Construct a HerbPairIAM model with the frozen config.

    Thin convenience wrapper. Internally dispatches to
    ``models.neural_models.build_model`` so that the constructed module is
    bit-identical to calling ``build_model("HerbPairIAM", ...)`` directly.
    """

    from models.neural_models import ModelConfig, build_model

    if config is None:
        cfg = ModelConfig(
            hidden=FROZEN_CONFIG["hidden"],
            dropout=FROZEN_CONFIG["dropout"],
            lr=FROZEN_CONFIG["lr"],
            epochs=FROZEN_CONFIG["epochs"],
            patience=FROZEN_CONFIG["patience"],
            batch_size=FROZEN_CONFIG["batch_size"],
            eval_every=FROZEN_CONFIG["eval_every"],
            neg_ratio=FROZEN_CONFIG["neg_ratio"],
        )
    elif isinstance(config, dict):
        cfg = ModelConfig(**config)
    else:
        cfg = config
    return build_model(MODEL_NAME, cfg, sample_example)
