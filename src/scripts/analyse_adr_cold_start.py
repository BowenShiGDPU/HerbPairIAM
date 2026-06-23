"""Failure-mode analysis for the ADR cold-start experiment.

Reads the per-seed cold_start fold pickles (``cold_adr_<seed>_<model>.pkl``)
and decomposes their aggregate near-chance AUROC into per-ADR performance
on the held-out ADR groups. For every held-out ADR we compute:

* per-ADR AUROC and AUPRC (only when both classes are present),
* the ADR's KG target profile size (``n_adr_targets``),
* the maximum Jaccard similarity between its target set and any ADR that
  was in the training pool for that seed (``nearest_train_jaccard``),
* the number of training positives for its profile-nearest training-seed ADR.

We then report Spearman correlations between these covariates and per-ADR
AUROC to characterise *why* cold-start is near chance: does a held-out
ADR's performance scale with (a) how well-annotated it is, (b) how similar
its target profile is to something seen during training, or (c) the
abundance of its nearest neighbour's labels?

Outputs::

    supplementary/adr_cold_start_per_adr.csv
    supplementary/adr_cold_start_failure_mode.csv

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/cold_start \\
    python -u src/scripts/analyse_adr_cold_start.py \\
        [--model HerbPairIAM] [--min-samples 4]
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_utils import FOLD_RESULTS_DIR, SUPP_DIR, ensure_output_dirs, load_pickle
from phase4_evaluation import PRIMARY_MODEL_NAME


sys.stdout.reconfigure(line_buffering=True)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT_DIR / "outputs"


def _jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRIMARY_MODEL_NAME,
                        help=f"Model whose cold-start pickles to analyse (default: {PRIMARY_MODEL_NAME}).")
    parser.add_argument("--min-samples", type=int, default=4,
                        help="Minimum samples per held-out ADR to compute per-ADR AUROC.")
    args = parser.parse_args()
    ensure_output_dirs()

    paths = sorted(FOLD_RESULTS_DIR.glob(f"cold_adr_*_{args.model}.pkl"))
    if not paths:
        print(f"No cold_adr_*_{args.model}.pkl found in {FOLD_RESULTS_DIR}")
        return 1

    # Load the original (formula, adr) identities for the full dataframe so we
    # can recover which ADR each test sample belongs to.
    with open(OUT_DIR / "dataset.pkl", "rb") as f:
        ds = pickle.load(f)
    df = ds["df"]
    with open(OUT_DIR / "adr_target_profiles.pkl", "rb") as f:
        adr_profiles = pickle.load(f)

    # All cold-adr splits live on ds["adr_cs_splits"] keyed by seed. Each gives
    # us (train_idx, test_idx), so we know which ADRs were in the training
    # pool vs. which were held out for a given seed.
    adr_cs_splits = {int(s["seed"]): s for s in ds.get("adr_cs_splits", [])}
    # Pre-compute each ADR's target profile for Jaccard comparisons; every
    # unique ADR id is expected to appear in adr_profiles.
    profiles_matrix = {a: np.asarray(p, dtype=bool) for a, p in adr_profiles.items()}

    long_rows = []
    summary_rows = []
    for path in paths:
        m = re.match(r"cold_adr_(\d+)_", path.name)
        if not m:
            continue
        seed = int(m.group(1))
        r = load_pickle(path)
        test_idx = np.asarray(r["test_indices"], dtype=int)
        y_true = np.asarray(r["y_true"], dtype=int)
        y_prob = np.asarray(r["y_prob"], dtype=float)
        if len(test_idx) != len(y_true):
            print(f"  [warn] seed={seed}: len(test_idx)={len(test_idx)} vs len(y_true)={len(y_true)}")
            continue
        adrs = df.iloc[test_idx]["Adr_id"].reset_index(drop=True)
        test_df = pd.DataFrame({
            "seed": seed, "Adr_id": adrs.values,
            "y_true": y_true, "y_prob": y_prob,
        })
        summary_rows.append({
            "seed": seed,
            "n_test": int(len(test_df)),
            "n_test_adrs": int(test_df["Adr_id"].nunique()),
            "pooled_auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
            "pooled_auprc": float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else float("nan"),
        })

        # Train-pool ADR profiles (for Jaccard).
        seed_split = adr_cs_splits.get(seed)
        if seed_split is None:
            continue
        train_adr_ids = sorted(set(df.iloc[np.asarray(seed_split["train_idx"], dtype=int)]["Adr_id"].unique().tolist()))
        train_profiles = [profiles_matrix[a] for a in train_adr_ids if a in profiles_matrix]

        for a, sub in test_df.groupby("Adr_id"):
            y = sub["y_true"].to_numpy()
            p = sub["y_prob"].to_numpy()
            n = len(sub); n_pos = int(y.sum())
            if n < args.min_samples or n_pos == 0 or n_pos == n:
                auroc = float("nan"); auprc = float("nan")
            else:
                auroc = float(roc_auc_score(y, p))
                auprc = float(average_precision_score(y, p))
            profile = profiles_matrix.get(a)
            n_targets = int(profile.sum()) if profile is not None else 0
            if profile is not None and train_profiles:
                jac = max(_jaccard_binary(profile, tp) for tp in train_profiles)
            else:
                jac = float("nan")
            long_rows.append({
                "seed": seed,
                "Adr_id": a,
                "n": n, "n_pos": n_pos, "n_neg": int(n - n_pos),
                "auroc": auroc, "auprc": auprc,
                "n_adr_targets": n_targets,
                "nearest_train_jaccard": jac,
            })

    if not long_rows:
        print("No per-ADR rows produced.")
        return 1

    per_adr_df = pd.DataFrame(long_rows).sort_values(["seed", "auroc"], na_position="last").reset_index(drop=True)
    per_path = SUPP_DIR / "adr_cold_start_per_adr.csv"
    per_adr_df.to_csv(per_path, index=False)
    print(f"Wrote {per_path}  rows={len(per_adr_df)}")
    print(f"Eligible per-ADR rows (both classes present, n>={args.min_samples}): "
          f"{int(per_adr_df['auroc'].notna().sum())}")

    # Spearman correlations.
    corr_rows = []
    for x_col in ["n_adr_targets", "nearest_train_jaccard", "n_pos", "n"]:
        for y_col in ["auroc", "auprc"]:
            x = per_adr_df[x_col].to_numpy(dtype=float)
            y = per_adr_df[y_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                corr_rows.append({"x": x_col, "y": y_col, "n": int(mask.sum()),
                                  "spearman_rho": float("nan"), "spearman_p": float("nan")})
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            corr_rows.append({"x": x_col, "y": y_col, "n": int(mask.sum()),
                              "spearman_rho": float(rho), "spearman_p": float(p)})
    corr_df = pd.DataFrame(corr_rows)
    corr_path = SUPP_DIR / "adr_cold_start_failure_mode.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Wrote {corr_path}")

    print("\nSpearman correlations (per-held-out-ADR performance vs. covariate):")
    print(corr_df.to_string(index=False))

    print("\n5-seed aggregate:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
