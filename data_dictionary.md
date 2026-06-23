# Data dictionary (processed inputs; full schema in the Zenodo archive)

- Entities: formula (multi-component Kampo product), crude-drug component, ingredient, compound, protein target, ADR (MedDRA preferred term).
- Signals: high-confidence formula-ADR associations curated from JADER and FAERS by a four-estimator disproportionality vote.
- `dataset.pkl`: candidate (formula, ADR) pairs with labels, feature columns, and leakage-controlled CV splits.
- Profiles: target-centered component/ADR/pair feature tables.
See `model_configurations.csv` for feature dimensions and dose handling.
