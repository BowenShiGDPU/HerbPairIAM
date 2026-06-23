"""Aggregate alliance-ablation fold pickles into ``alliance_ablation.csv``.

Computes paired t-tests of every leave-one-alliance-out setting against
``full`` on the same 10 fold splits and applies Holm-Bonferroni correction
across the four ablation comparisons.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/alliance_ablation \\
    python -u src/scripts/aggregate_formal_alliance_ablation.py
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import sys

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import (
    FOLD_RESULTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    holm_bonferroni,
    load_pickle,
)
from neural_models import summarize_results as summarize_neural, model_intrinsic_ablation
from phase4_evaluation import FORMAL_ALLIANCE_ABLATIONS, PRIMARY_MODEL_NAME


sys.stdout.reconfigure(line_buffering=True)


def _load_setting_results(setting: str) -> list[dict]:
    tag = f"{PRIMARY_MODEL_NAME}__{setting}"
    results = []
    for fold_id in range(10):
        path = FOLD_RESULTS_DIR / f"{tag}_fold{fold_id}.pkl"
        if not path.exists():
            continue
        try:
            results.append(load_pickle(path))
        except Exception:
            continue
    return sorted(results, key=lambda item: int(item.get("fold", 0)))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_dirs()

    # Tags that the primary model already applies to its samples. Settings
    # whose alliance tag set is a subset of these are structurally equivalent
    # to ``full`` for this model and will be flagged as such in the CSV.
    intrinsic_tags = model_intrinsic_ablation(PRIMARY_MODEL_NAME)

    rows = []
    fold_results: dict[str, list[dict]] = {}
    for setting, tag_set in FORMAL_ALLIANCE_ABLATIONS:
        results = _load_setting_results(setting)
        if not results:
            print(f"  [skip] no fold pkl for setting={setting}")
            continue
        fold_results[setting] = results
        summary = summarize_neural(results)
        rows.append(
            {
                "Model": PRIMARY_MODEL_NAME,
                "Setting": setting,
                "n_folds": int(len(results)),
                "AUROC": float(summary["auroc_mean"]),
                "AUPRC": float(summary["auprc_mean"]),
                "AUROC_std": float(summary["auroc_std"]),
                "AUPRC_std": float(summary["auprc_std"]),
                # True when the alliance tag set is already intrinsic to the
                # primary model (e.g. AL_dose for HerbPairIAM); in that case
                # the "ablation" does not perturb the training data and the
                # result must equal the full reference.
                "intrinsic_to_primary": bool(
                    tag_set and set(tag_set).issubset(intrinsic_tags)
                ),
            }
        )

    if not rows:
        print(f"No alliance-ablation fold pkls found in {FOLD_RESULTS_DIR}")
        return 1

    full_row = next((r for r in rows if r["Setting"] == "full"), None)
    full_results = fold_results.get("full") if full_row else None
    if full_row is None or full_results is None:
        print("WARN: full setting missing; skipping paired stats")
    else:
        full_auroc_per_fold = np.asarray([float(r["auroc"]) for r in full_results], dtype=float)
        full_auprc_per_fold = np.asarray([float(r["auprc"]) for r in full_results], dtype=float)

        pvals_auroc: dict[str, float] = {}
        pvals_auprc: dict[str, float] = {}
        for r in rows:
            setting = r["Setting"]
            r["delta_AUROC_vs_full"] = float(r["AUROC"] - full_row["AUROC"])
            r["delta_AUPRC_vs_full"] = float(r["AUPRC"] - full_row["AUPRC"])
            if setting == "full":
                r["p_AUROC_vs_full"] = float("nan")
                r["p_AUPRC_vs_full"] = float("nan")
                continue
            sib = fold_results.get(setting, [])
            if len(sib) != len(full_results):
                r["p_AUROC_vs_full"] = float("nan")
                r["p_AUPRC_vs_full"] = float("nan")
                continue
            sib_auroc = np.asarray([float(x["auroc"]) for x in sib], dtype=float)
            sib_auprc = np.asarray([float(x["auprc"]) for x in sib], dtype=float)
            try:
                _, p1 = stats.ttest_rel(full_auroc_per_fold, sib_auroc)
            except Exception:
                p1 = float("nan")
            try:
                _, p2 = stats.ttest_rel(full_auprc_per_fold, sib_auprc)
            except Exception:
                p2 = float("nan")
            r["p_AUROC_vs_full"] = float(p1)
            r["p_AUPRC_vs_full"] = float(p2)
            if not np.isnan(p1):
                pvals_auroc[setting] = float(p1)
            if not np.isnan(p2):
                pvals_auprc[setting] = float(p2)

        holm_a = holm_bonferroni(pvals_auroc)
        holm_b = holm_bonferroni(pvals_auprc)
        for r in rows:
            setting = r["Setting"]
            r["pHolm_AUROC_vs_full"] = holm_a.get(setting) if setting != "full" else None
            r["pHolm_AUPRC_vs_full"] = holm_b.get(setting) if setting != "full" else None

    out_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "alliance_ablation.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
