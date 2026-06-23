"""Pair-wise DeLong test for pooled OOF AUROC.

For every ordered pair of models with complete fold pickles under
``EXPERIMENT_SUBDIR/fold_results/``, we compute the DeLong
(1988 / fastDeLong 2014) two-sided p-value on the pooled concatenated
(y_true, y_prob) vectors. DeLong is the standard non-parametric test for
comparing two correlated ROC curves at the *sample* level; it is what
clinical-ML journals expect when two models are scored on the same
evaluation sample set.

The script writes::

    supplementary/delong_pairwise.csv          # long-format (row per pair)
    supplementary/delong_pvalue_matrix.csv     # square p-value matrix
    supplementary/delong_holm_adjusted.csv     # long-format with Holm

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/scripts/compute_pairwise_delong.py
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from experiment_utils import (
    FOLD_RESULTS_DIR,
    SUPP_DIR,
    delong_test,
    ensure_output_dirs,
    holm_bonferroni,
    load_pickle,
)


sys.stdout.reconfigure(line_buffering=True)


_FOLD_RE = re.compile(r"^(?P<model>.+)_fold\d+$")


def _load_by_filename_model(min_folds: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not FOLD_RESULTS_DIR.exists():
        return out
    for p in sorted(FOLD_RESULTS_DIR.glob("*_fold*.pkl")):
        m = _FOLD_RE.match(p.stem)
        if not m:
            continue
        try:
            obj = load_pickle(p)
        except Exception:
            continue
        if "y_true" not in obj or "y_prob" not in obj:
            continue
        out.setdefault(m.group("model"), []).append(obj)
    for k, rs in out.items():
        out[k] = sorted(rs, key=lambda r: int(r.get("fold", 0)))
    return {k: rs for k, rs in out.items() if len(rs) >= min_folds}


def _pool(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y_t = np.concatenate([np.asarray(r["y_true"], dtype=int) for r in results])
    y_p = np.concatenate([np.asarray(r["y_prob"], dtype=float) for r in results])
    return y_t, y_p


def _assert_same_y_true(models: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """DeLong is only valid when both models score the same samples.

    We verify all models share an identical pooled y_true; this holds for our
    10-fold CV because every model's OOF test sets are the same 10 folds.
    Hard-linked / renamed pickles can break this invariant, so the check is
    not optional.
    """
    names = list(models)
    ref = models[names[0]][0]
    for n in names[1:]:
        y_t, _ = models[n]
        if y_t.shape != ref.shape or not np.array_equal(y_t, ref):
            raise ValueError(
                f"Pooled y_true for {n!r} does not match {names[0]!r}; "
                "DeLong requires the two models to be scored on the same samples."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-folds", type=int, default=10,
                        help="Minimum fold pickles required to include a model (default 10).")
    args = parser.parse_args()
    ensure_output_dirs()

    raw = _load_by_filename_model(min_folds=args.min_folds)
    if len(raw) < 2:
        print(f"Need at least two models with >= {args.min_folds} folds; found {list(raw)}")
        return 1

    pooled = {m: _pool(rs) for m, rs in raw.items()}
    _assert_same_y_true(pooled)
    print(f"Computing DeLong p-values across {len(pooled)} models...")

    model_names = sorted(pooled)
    pair_rows = []
    pval_lookup: dict[str, float] = {}
    for a, b in combinations(model_names, 2):
        y_true, y_a = pooled[a]
        _,      y_b = pooled[b]
        # Two-sided DeLong.
        p = delong_test(y_true, y_a, y_b)
        from sklearn.metrics import roc_auc_score
        auc_a = float(roc_auc_score(y_true, y_a))
        auc_b = float(roc_auc_score(y_true, y_b))
        pair_rows.append({
            "model_a": a,
            "model_b": b,
            "auroc_a": auc_a,
            "auroc_b": auc_b,
            "delta_auroc": auc_a - auc_b,
            "delong_p": float(p),
        })
        pval_lookup[f"{a}__vs__{b}"] = float(p)

    df = pd.DataFrame(pair_rows).sort_values("delong_p").reset_index(drop=True)
    df.to_csv(SUPP_DIR / "delong_pairwise.csv", index=False)
    print(f"Wrote {SUPP_DIR / 'delong_pairwise.csv'}  rows={len(df)}")

    # Holm-Bonferroni across all C(n,2) comparisons.
    p_adj = holm_bonferroni(pval_lookup)
    df_adj = df.copy()
    df_adj["delong_p_holm"] = [p_adj[f"{r.model_a}__vs__{r.model_b}"] for r in df.itertuples()]
    df_adj.to_csv(SUPP_DIR / "delong_holm_adjusted.csv", index=False)
    print(f"Wrote {SUPP_DIR / 'delong_holm_adjusted.csv'}  rows={len(df_adj)}")

    # Square matrix (symmetric, NaN on the diagonal).
    mat = pd.DataFrame(np.nan, index=model_names, columns=model_names, dtype=float)
    for r in pair_rows:
        mat.loc[r["model_a"], r["model_b"]] = r["delong_p"]
        mat.loc[r["model_b"], r["model_a"]] = r["delong_p"]
    mat.to_csv(SUPP_DIR / "delong_pvalue_matrix.csv")
    print(f"Wrote {SUPP_DIR / 'delong_pvalue_matrix.csv'}")

    # Compact stdout preview of the top comparisons.
    print("\nTop comparisons (smallest Holm-adjusted p-value):")
    print(df_adj.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
