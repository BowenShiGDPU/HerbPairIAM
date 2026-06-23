# Model naming

This note maps the paper-facing model name to the class names used in the source
code, so the repository can be read alongside the manuscript without ambiguity.

## Primary model

**`HerbPairIAM`** is the paper-facing name of the primary model; all headline
results (main benchmark, ablations, cold-start, interpretability) are reported on
it. In code it is the **`DoseAwareInteractionAwareSetModel`** class
(`src/models/neural_models.py`) instantiated with its dose-input channels
**zero-filled**. The model-construction factory treats `HerbPairIAM` and the
internal alias `DoseAware_ZeroDose` as the same configuration; `HerbPairIAM` is the
canonical entry point (`PRIMARY_MODEL_NAME`; see `src/models/herb_pair_iam.py`).

## Internal architecture-variant names — *not* public baselines

The names below are internal implementation / design-check labels. They exist only
to support the dose-design ablation that motivates the zero-dose choice
(`src/scripts/run_dose_head2head.py`) and should **not** be read as manuscript
comparator baselines:

| Code name | What it is |
|---|---|
| `DoseAwareIAM` | The same architecture as HerbPairIAM but fed **real** dose values — the real-dose comparator that the zero-dose design is tested against. |
| `InteractionAwareSetModel` (`IAM`) | The set model with **no** dose pathway. |
| `IAM_Wide` | `InteractionAwareSetModel` widened (hidden=44) to capacity-match the dose-aware model, separating architecture from parameter count. |

## Public comparator baselines (paper-facing)

The paper's comparators are seven standard baselines, implemented in
`src/models/tabular_models.py` and `src/models/graph_baselines.py`:
Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP, R-GCN, HGT.
