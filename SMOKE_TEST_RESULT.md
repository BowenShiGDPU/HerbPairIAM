# Release validation — clean-checkout smoke test

This package was validated from a fresh checkout outside its development tree:

- Every script byte-compiles: `python -m compileall src` succeeds for all 85 modules (0 errors, 0 warnings).
- The primary model (`HerbPairIAM`) trains end-to-end on canonical fold 0 using only repository-root-relative paths — a reduced-epoch run, not intended to reproduce the reported score:

      SMOKE_OK fold0 reduced(3 epochs): AUROC=0.8231 AUPRC=0.5234 n_test=397 elapsed=33.2s

- Validation environment: Python 3.12 with torch 2.11.0, torch-geometric 2.7.0, xgboost 3.2.0, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1.
- No machine-specific paths are hardcoded; all inputs/outputs resolve relative to the repository root (`outputs/`, `final_data_clean/`).
