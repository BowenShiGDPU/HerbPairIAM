"""HerbPairIAM: the primary model.

HerbPairIAM is an interaction-aware attention model that scores
formula-ADR pairs over a pharmacovigilance knowledge graph. It shares the
``DoseAwareInteractionAwareSetModel`` architecture defined in
``models.neural_models`` but is always fed zero-filled dose inputs
(enforced sample-level ablation ``AL_dose``), so the auxiliary dose
branch effectively becomes a learned constant bias channel.

Build the model with::

    from models.neural_models import build_model, ModelConfig
    model = build_model("HerbPairIAM", ModelConfig(), sample_example)

Zero-dose samples are produced by::

    from models.neural_models import build_sample_collections
    sample_map = build_sample_collections(df, lookups, hp, ap, pf,
                                          ["HerbPairIAM"])

since ``model_intrinsic_ablation("HerbPairIAM") == frozenset({"AL_dose"})``.
"""

from __future__ import annotations

MODEL_NAME = "HerbPairIAM"
ALIASES = ("DoseAware_ZeroDose",)
INTRINSIC_ABLATION_TAGS = frozenset({"AL_dose"})

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
    """Construct a HerbPairIAM model with the frozen config."""

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
