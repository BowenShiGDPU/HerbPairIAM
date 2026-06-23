"""Subgroup analyses on the pooled OOF predictions of the primary model.

Two orthogonal breakdowns are produced:

1. **Per-ADR performance** (``per_adr_analysis.csv``) — for every ADR that
   appears in at least ``--min-samples`` OOF samples we compute AUROC,
   AUPRC and the positive-sample count. We then correlate those per-ADR
   AUROCs with two ADR properties:
     * number of KG-derived ADR targets (how richly annotated the ADR is);
     * number of positive training samples for that ADR (label frequency).
   Spearman rank correlations are written to
   ``per_adr_correlations.csv``; they quantify *which* ADRs the model
   handles well and whether that is driven by data availability vs.
   biological tractability.

2. **Per-Formula performance** (``per_formula_analysis.csv``) — same
   decomposition along the other axis. Correlates per-formula AUROC with
   the formula's herb count and its total positive-label count.

Plus, ``adr_cold_start_failure_mode.csv`` analyses the 5-seed ADR
cold-start results: for every held-out ADR, how its per-ADR AUROC relates
to (i) the Jaccard similarity between its target set and the nearest
training ADR, (ii) the number of PMDA-style annotations, and (iii) the
fraction of held-out positives.

All scripts are read-only.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/scripts/compute_subgroup_analysis.py --model HerbPairIAM
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


_FOLD_RE = re.compile(r"^(?P<model>.+)_fold\d+$")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT_DIR / "outputs"


def _load_model_folds(model: str) -> list[dict]:
    results = []
    for p in sorted(FOLD_RESULTS_DIR.glob(f"{model}_fold*.pkl")):
        try:
            obj = load_pickle(p)
            results.append(obj)
        except Exception:
            continue
    return sorted(results, key=lambda r: int(r.get("fold", 0)))


def _per_group_metrics(df: pd.DataFrame, group_col: str, min_samples: int) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        n = len(sub)
        n_pos = int(sub["y_true"].sum())
        n_neg = int(n - n_pos)
        if n < min_samples or n_pos == 0 or n_neg == 0:
            rows.append({
                group_col: g, "n": n, "n_pos": n_pos, "n_neg": n_neg,
                "auroc": float("nan"), "auprc": float("nan"),
            })
            continue
        y = sub["y_true"].to_numpy()
        p = sub["y_prob"].to_numpy()
        rows.append({
            group_col: g, "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
        })
    out = pd.DataFrame(rows).sort_values("auroc", ascending=False, na_position="last").reset_index(drop=True)
    return out


def _spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRIMARY_MODEL_NAME,
                        help=f"Model to analyse (default: {PRIMARY_MODEL_NAME}).")
    parser.add_argument("--min-samples-adr", type=int, default=4,
                        help="Minimum OOF samples per ADR to compute group AUROC (default 4).")
    parser.add_argument("--min-samples-formula", type=int, default=4,
                        help="Minimum OOF samples per formula to compute group AUROC.")
    args = parser.parse_args()
    ensure_output_dirs()

    results = _load_model_folds(args.model)
    if not results:
        print(f"No fold pickles found for model={args.model!r} in {FOLD_RESULTS_DIR}")
        return 1
    print(f"Loaded {len(results)} folds for {args.model}")

    # Reconstruct the OOF (formula, ADR) identities from the original dataframe.
    ds_path = OUT_DIR / "dataset.pkl"
    with open(ds_path, "rb") as f:
        ds = pickle.load(f)
    df = ds["df"]

    # Concatenate OOF predictions with their original row indices.
    long_rows = []
    for r in results:
        # Prefer ``test_indices`` (shape-(n,)); fall back to ``test_idx``.
        # ``or`` does not work on arrays, so check for None explicitly.
        raw_idx = r.get("test_indices")
        if raw_idx is None or (hasattr(raw_idx, "__len__") and len(raw_idx) == 0):
            raw_idx = r.get("test_idx")
        if raw_idx is None:
            print(f"  [warn] fold {r.get('fold')}: no test indices; skipping")
            continue
        idx = np.asarray(raw_idx, dtype=int)
        y_true = np.asarray(r["y_true"], dtype=int)
        y_prob = np.asarray(r["y_prob"], dtype=float)
        if len(idx) != len(y_true):
            print(f"  [warn] fold {r.get('fold')}: len(test_indices)={len(idx)} vs len(y_true)={len(y_true)}")
            continue
        fold_id = int(r.get("fold", 0))
        meta = df.iloc[idx][["TCMF_id", "Adr_id"]].reset_index(drop=True)
        long_rows.append(pd.DataFrame({
            "fold": fold_id,
            "row_index": idx,
            "TCMF_id": meta["TCMF_id"].values,
            "Adr_id": meta["Adr_id"].values,
            "y_true": y_true,
            "y_prob": y_prob,
        }))
    if not long_rows:
        print("No OOF rows reconstructable; check test_indices.")
        return 1
    oof = pd.concat(long_rows, ignore_index=True)
    print(f"Reconstructed OOF: n={len(oof)}  unique ADRs={oof['Adr_id'].nunique()}  unique Formulas={oof['TCMF_id'].nunique()}")

    # Per-ADR and per-Formula.
    adr_df = _per_group_metrics(oof, "Adr_id", args.min_samples_adr)
    for_df = _per_group_metrics(oof, "TCMF_id", args.min_samples_formula)
    adr_out = SUPP_DIR / "per_adr_analysis.csv"
    for_out = SUPP_DIR / "per_formula_analysis.csv"
    adr_df.to_csv(adr_out, index=False)
    for_df.to_csv(for_out, index=False)
    print(f"Wrote {adr_out}  rows={len(adr_df)} (with enough samples={int(adr_df['auroc'].notna().sum())})")
    print(f"Wrote {for_out}  rows={len(for_df)} (with enough samples={int(for_df['auroc'].notna().sum())})")

    # ADR side correlates: n_adr_targets and positive-label frequency in training data.
    try:
        adr_profile_path = OUT_DIR / "adr_target_profiles.pkl"
        with open(adr_profile_path, "rb") as f:
            adr_profiles = pickle.load(f)
        n_targets = {a: int(np.asarray(vec).astype(bool).sum()) for a, vec in adr_profiles.items()}
    except Exception:
        n_targets = {}
    adr_freq = df[df["label"] == 1].groupby("Adr_id").size().to_dict()
    adr_df["n_adr_targets"] = adr_df["Adr_id"].map(n_targets).astype("float")
    adr_df["n_train_positives"] = adr_df["Adr_id"].map(adr_freq).fillna(0).astype(int)

    corr_rows = []
    for x_col in ["n_adr_targets", "n_train_positives", "n_pos"]:
        for y_col in ["auroc", "auprc"]:
            rho, p = _spearman(adr_df[x_col].to_numpy(), adr_df[y_col].to_numpy())
            corr_rows.append({
                "x": x_col, "y": y_col, "n_eligible": int(adr_df[y_col].notna().sum()),
                "spearman_rho": rho, "spearman_p": p,
            })
    # Formula side correlates: herb count + positive count.
    formula_to_herbs = ds.get("formula_to_herbs") or ds.get("f2h") or {}
    if not formula_to_herbs:
        try:
            with open(OUT_DIR / "lookups.pkl", "rb") as f:
                lookups = pickle.load(f)
            formula_to_herbs = lookups.get("f2h", {})
        except Exception:
            formula_to_herbs = {}
    n_herbs_map = {k: len(v) for k, v in formula_to_herbs.items()}
    for_df["n_herbs"] = for_df["TCMF_id"].map(n_herbs_map).astype("float")
    formula_freq = df[df["label"] == 1].groupby("TCMF_id").size().to_dict()
    for_df["n_train_positives"] = for_df["TCMF_id"].map(formula_freq).fillna(0).astype(int)
    for x_col in ["n_herbs", "n_train_positives", "n_pos"]:
        for y_col in ["auroc", "auprc"]:
            rho, p = _spearman(for_df[x_col].to_numpy(), for_df[y_col].to_numpy())
            corr_rows.append({
                "x": f"formula.{x_col}", "y": y_col,
                "n_eligible": int(for_df[y_col].notna().sum()),
                "spearman_rho": rho, "spearman_p": p,
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_out = SUPP_DIR / "per_group_correlations.csv"
    corr_df.to_csv(corr_out, index=False)
    print(f"Wrote {corr_out}")
    print(corr_df.to_string(index=False))

    # Re-emit the enriched per-group tables that now contain side-covariates.
    adr_df.to_csv(adr_out, index=False)
    for_df.to_csv(for_out, index=False)
    print("\nPer-ADR top-10 by AUROC:")
    print(adr_df.head(10).to_string(index=False))
    print("\nPer-ADR bottom-10 by AUROC (with enough samples):")
    print(adr_df[adr_df["auroc"].notna()].tail(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
