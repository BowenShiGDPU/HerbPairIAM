"""Leave-one-herb counterfactual analysis for attention-as-explanation validity.

For each OOF test sample of the primary model we produce:
  1. the model's predicted probability on the full sample (``prob_full``),
  2. the predicted probability when the i-th herb is removed from the
     sample (``prob_lo_i``),
  3. the herb-attention weight the model assigned to the i-th herb
     (``attn_i``).

We then test whether ``attn_i`` is positively correlated with
``|prob_full − prob_lo_i|`` (the absolute predictive impact of removing
herb i). A significant positive Spearman correlation is the standard
sanity check for "attention reflects contribution" in interpretability
papers (Serrano & Smith 2019, Wiegreffe & Pinter 2019).

Method
------
Removing the i-th herb from a sample entails:
  * dropping its row from ``node_features``,
  * dropping every pair that includes it (both the row and the new
    ``pair_src / pair_dst`` indices need to reflect the remaining
    C(n-1, 2) pairs in the original ordering),
  * dropping the corresponding entries in ``pair_features``.

We keep everything else (ADR features, zero-dose tails, attention
softmax mechanism) identical. No retraining: we reload the canonical
state dict per fold and run forward on each counterfactual sample.

Only samples with at least 2 herbs are considered (removing the single
herb of a 1-herb formula is degenerate). A sample's contribution is the
vector of (attn_i, |Δprob_i|) for every herb i in that sample.

Outputs::

    supplementary/leave_one_herb_counterfactual.csv          # per (fold, sample, herb) rows
    supplementary/leave_one_herb_counterfactual_summary.csv  # correlations + per-fold Spearman

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/scripts/compute_leave_one_herb_counterfactual.py \\
        [--max-samples -1] [--positives-only]
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from experiment_utils import FOLD_RESULTS_DIR, MODELS_DIR, SUPP_DIR, ensure_output_dirs
from models.neural_models import (
    DEVICE,
    ModelConfig,
    build_model,
    build_sample_collections,
    load_all,
)
from phase4_evaluation import PRIMARY_MODEL_NAME


sys.stdout.reconfigure(line_buffering=True)


def _drop_herb_sample(sample: dict, drop_i: int) -> dict:
    """Return a new sample dict with the ``drop_i``-th herb removed.

    The node / pair tensors and pair-src / pair-dst indices are rebuilt so
    they refer to the C(n-1, 2) pairs of the reduced herb set. All other
    fields are passed through unchanged.
    """
    n = int(sample["node_features"].shape[0])
    assert 0 <= drop_i < n
    keep = [i for i in range(n) if i != drop_i]

    # Node-side tensors.
    node_features = sample["node_features"][keep]

    # Pair-side: rebuild pair_src / pair_dst / pair_features so the remaining
    # pairs reference positions 0..n-2 in the reduced herb list.
    pair_src_old = sample["pair_src"].cpu().numpy() if sample["pair_src"].numel() > 0 else np.empty(0, dtype=int)
    pair_dst_old = sample["pair_dst"].cpu().numpy() if sample["pair_dst"].numel() > 0 else np.empty(0, dtype=int)
    mask = (pair_src_old != drop_i) & (pair_dst_old != drop_i)
    new_index = {old: new for new, old in enumerate(keep)}
    new_pair_src = np.asarray([new_index[i] for i in pair_src_old[mask]], dtype=np.int64)
    new_pair_dst = np.asarray([new_index[i] for i in pair_dst_old[mask]], dtype=np.int64)
    pair_features = sample["pair_features"][mask]

    # edge_src / edge_dst / edge_features use the same index space as node_*
    # with both directions emitted. Drop any edge touching drop_i and reindex.
    edge_src_old = sample["edge_src"].cpu().numpy() if sample["edge_src"].numel() > 0 else np.empty(0, dtype=int)
    edge_dst_old = sample["edge_dst"].cpu().numpy() if sample["edge_dst"].numel() > 0 else np.empty(0, dtype=int)
    emask = (edge_src_old != drop_i) & (edge_dst_old != drop_i)
    new_edge_src = np.asarray([new_index[i] for i in edge_src_old[emask]], dtype=np.int64)
    new_edge_dst = np.asarray([new_index[i] for i in edge_dst_old[emask]], dtype=np.int64)
    edge_features = sample["edge_features"][emask] if sample["edge_features"].numel() > 0 else sample["edge_features"]

    new_sample = dict(sample)
    new_sample["node_features"] = node_features
    new_sample["pair_src"] = torch.tensor(new_pair_src, dtype=torch.long, device=DEVICE)
    new_sample["pair_dst"] = torch.tensor(new_pair_dst, dtype=torch.long, device=DEVICE)
    new_sample["pair_features"] = pair_features
    new_sample["edge_src"] = torch.tensor(new_edge_src, dtype=torch.long, device=DEVICE)
    new_sample["edge_dst"] = torch.tensor(new_edge_dst, dtype=torch.long, device=DEVICE)
    new_sample["edge_features"] = edge_features
    new_sample["n_herbs"] = len(keep)
    new_sample["herbs"] = [sample["herbs"][i] for i in keep]
    return new_sample


@torch.no_grad()
def _sigmoid(x: torch.Tensor) -> float:
    return float(torch.sigmoid(x).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRIMARY_MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Debug: limit to the first N test samples per fold (−1 = all).")
    parser.add_argument("--positives-only", action="store_true",
                        help="Restrict to label=1 samples to focus on driver-herb contributions.")
    args = parser.parse_args()
    ensure_output_dirs()

    ds, hp, ap, pf, lookups = load_all()
    df = ds["df"].reset_index(drop=True)
    sample_map = build_sample_collections(df, lookups, hp, ap, pf, [args.model])
    samples = sample_map[args.model]

    sample_example = next((s for s in samples if s is not None), None)
    cfg = ModelConfig(hidden=32, dropout=0.3, lr=1e-3, epochs=100, patience=10, batch_size=32, neg_ratio=10, eval_every=2)

    per_herb_rows = []
    per_sample_rows = []
    fold_summaries = []

    for split in ds["fold_splits"]:
        fold_id = int(split["fold"])
        state_path = MODELS_DIR / f"{args.model}_fold{fold_id}.pt"
        if not state_path.exists():
            print(f"  [skip] fold {fold_id}: no state dict at {state_path}")
            continue
        model = build_model(args.model, cfg, sample_example=sample_example)
        state = torch.load(state_path, map_location=DEVICE)
        info = model.load_state_dict(state, strict=False)
        if info.missing_keys or info.unexpected_keys:
            print(f"  [fold {fold_id}] non-strict load: missing={list(info.missing_keys)[:3]} "
                  f"unexpected={list(info.unexpected_keys)[:3]}")
        model.eval()

        test_idx = split["test_idx"]
        n_considered = 0
        fold_pairs: list[tuple[float, float]] = []
        for idx in test_idx:
            sample = samples[int(idx)]
            if sample is None:
                continue
            label = int(df.iloc[int(idx)]["label"])
            if args.positives_only and label != 1:
                continue
            n_herbs = int(sample["node_features"].shape[0])
            if n_herbs < 2:
                continue
            if args.max_samples > 0 and n_considered >= args.max_samples:
                break
            n_considered += 1

            # Full-sample forward.
            logit_full, aux_full = model(sample)
            prob_full = _sigmoid(logit_full)
            herb_attn = aux_full["herb_attn"].cpu().numpy()
            if herb_attn.shape[0] != n_herbs:
                # Mean-pool / non-attention models expose a degenerate attn
                # tensor; skip the sample rather than compare to a constant.
                continue

            delta_probs = np.zeros(n_herbs, dtype=float)
            for i in range(n_herbs):
                cf_sample = _drop_herb_sample(sample, i)
                logit_cf, _ = model(cf_sample)
                prob_cf = _sigmoid(logit_cf)
                delta_probs[i] = prob_full - prob_cf  # positive ⇒ herb i pushed prob up
                per_herb_rows.append({
                    "fold": fold_id,
                    "sample_idx": int(idx),
                    "label": label,
                    "herb_position": i,
                    "herb_id": sample["herbs"][i],
                    "attention": float(herb_attn[i]),
                    "prob_full": prob_full,
                    "prob_without_herb": prob_cf,
                    "delta_prob": float(delta_probs[i]),
                    "abs_delta_prob": float(abs(delta_probs[i])),
                })

            # Per-sample Spearman (only defined for n_herbs >= 3).
            if n_herbs >= 3:
                try:
                    rho_abs, p_abs = stats.spearmanr(herb_attn, np.abs(delta_probs))
                    rho_signed, p_signed = stats.spearmanr(herb_attn, delta_probs)
                except Exception:
                    rho_abs = p_abs = rho_signed = p_signed = float("nan")
                per_sample_rows.append({
                    "fold": fold_id,
                    "sample_idx": int(idx),
                    "label": label,
                    "n_herbs": n_herbs,
                    "spearman_attn_vs_abs_delta": float(rho_abs),
                    "spearman_attn_vs_signed_delta": float(rho_signed),
                    "spearman_p_abs": float(p_abs),
                    "spearman_p_signed": float(p_signed),
                })
                fold_pairs.append((rho_abs, p_abs))

        # Per-fold aggregate.
        if fold_pairs:
            rhos = np.array([r for r, _ in fold_pairs if np.isfinite(r)], dtype=float)
            fold_summaries.append({
                "fold": fold_id,
                "n_samples": len(fold_pairs),
                "mean_spearman_attn_abs_delta": float(rhos.mean()) if rhos.size else float("nan"),
                "median_spearman_attn_abs_delta": float(np.median(rhos)) if rhos.size else float("nan"),
                "frac_positive_rho": float((rhos > 0).mean()) if rhos.size else float("nan"),
            })
        print(f"  [fold {fold_id}] processed {n_considered} samples, "
              f"{len(fold_pairs)} with n_herbs>=3")

    # Global herb-level DataFrame.
    per_herb_df = pd.DataFrame(per_herb_rows)
    per_sample_df = pd.DataFrame(per_sample_rows)
    fold_df = pd.DataFrame(fold_summaries)

    out_a = SUPP_DIR / "leave_one_herb_counterfactual.csv"
    out_b = SUPP_DIR / "leave_one_herb_counterfactual_per_sample.csv"
    out_c = SUPP_DIR / "leave_one_herb_counterfactual_summary.csv"
    per_herb_df.to_csv(out_a, index=False)
    per_sample_df.to_csv(out_b, index=False)
    print(f"Wrote {out_a}  rows={len(per_herb_df)}")
    print(f"Wrote {out_b}  rows={len(per_sample_df)}")

    # Global Spearman across all (attention, |Δprob|) pairs.
    if len(per_herb_df) >= 3:
        rho_all, p_all = stats.spearmanr(per_herb_df["attention"], per_herb_df["abs_delta_prob"])
        rho_signed_all, p_signed_all = stats.spearmanr(per_herb_df["attention"], per_herb_df["delta_prob"])
    else:
        rho_all = p_all = rho_signed_all = p_signed_all = float("nan")

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.loc["__global__"] = {
        "fold": "global",
        "n_samples": int(per_sample_df.shape[0]),
        "mean_spearman_attn_abs_delta": float("nan"),
        "median_spearman_attn_abs_delta": float("nan"),
        "frac_positive_rho": float(
            (per_sample_df["spearman_attn_vs_abs_delta"] > 0).mean()
            if not per_sample_df.empty else float("nan")
        ),
    }
    fold_df.to_csv(out_c, index=False)
    print(f"Wrote {out_c}  rows={len(fold_df)}")

    # Stdout summary.
    print()
    print(f"=== Global pooled Spearman (all n={len(per_herb_df)} (herb, sample) pairs) ===")
    print(f"  attention vs |Δprob|:  rho = {rho_all:.4f}  p = {p_all:.3e}")
    print(f"  attention vs  Δprob :  rho = {rho_signed_all:.4f}  p = {p_signed_all:.3e}")
    if not per_sample_df.empty:
        rho_median = float(per_sample_df["spearman_attn_vs_abs_delta"].median())
        rho_mean   = float(per_sample_df["spearman_attn_vs_abs_delta"].mean())
        pos_frac   = float((per_sample_df["spearman_attn_vs_abs_delta"] > 0).mean())
        sig_frac   = float((per_sample_df["spearman_p_abs"] < 0.05).mean())
        print(f"\n=== Per-sample Spearman (n={len(per_sample_df)} samples with >= 3 herbs) ===")
        print(f"  median rho = {rho_median:.4f}   mean rho = {rho_mean:.4f}")
        print(f"  fraction of samples with rho > 0:  {pos_frac:.3f}")
        print(f"  fraction of samples with p < 0.05:  {sig_frac:.3f}")
    print(f"\nFiles:\n  {out_a}\n  {out_b}\n  {out_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
