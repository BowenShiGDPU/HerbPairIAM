"""Re-derive Main Table 3 (structure ablation) directly from fold pickles.

Loads all ablation pickles, (a) rebuilds per-fold AUROC/AUPRC using
``sklearn.metrics`` so results are independent of any cached CSV, and
(b) computes paired two-sided Wilcoxon signed-rank p-values against
HerbPairIAM on the same 10 folds. Holm-Bonferroni correction is
applied across the 9 non-reference variants for each metric.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = Path(__file__).resolve().parents[2] / "results" / "formal_doseaware_neg10_auroc"
MB = RESULTS / "main_benchmark" / "fold_results"
SA = RESULTS / "structure_ablation" / "fold_results"
H2H = RESULTS / "dose_head2head" / "fold_results"

VARIANT_ROWS = [
    # (display_label, model_tag, directory, fold_template)
    ("HerbPairIAM (primary)",                                           "HerbPairIAM",                MB,  "{name}_fold{k}.pkl"),
    ("real dose vector supplied on aux input",                          "DoseAwareIAM",               MB,  "{name}_fold{k}.pkl"),
    ("aux branch removed, hidden width unchanged",                      "InteractionAwareSetModel",   MB,  "{name}_fold{k}.pkl"),
    ("aux branch removed, hidden width expanded to match capacity",     "IAM_Wide",                   MB,  "{name}_fold{k}.pkl"),
    ("ADR-context gate removed",                                        "DoseAwareNoDoseGate",        SA,  "{name}_fold{k}.pkl"),
    ("pair branch removed",                                             "DoseAwareHerbOnly",          SA,  "{name}_fold{k}.pkl"),
    ("herb branch removed",                                             "DoseAwarePairOnly",          SA,  "{name}_fold{k}.pkl"),
    ("ADR-conditioned attention disabled",                              "DoseAwareNoADRConditioning", SA,  "{name}_fold{k}.pkl"),
    ("attention replaced by uniform mean pool",                         "DoseAwareMeanPool",          SA,  "{name}_fold{k}.pkl"),
]


def _per_fold(tag: str, directory: Path, tmpl: str) -> tuple[np.ndarray, np.ndarray]:
    aur, apr = [], []
    for k in range(10):
        fp = directory / tmpl.format(name=tag, k=k)
        with open(fp, "rb") as fh:
            r = pickle.load(fh)
        aur.append(roc_auc_score(r["y_true"], r["y_prob"]))
        apr.append(average_precision_score(r["y_true"], r["y_prob"]))
    return np.array(aur), np.array(apr)


def _holm(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, float)
    m = len(arr)
    order = np.argsort(arr)
    adj = np.ones_like(arr)
    running = 0.0
    for step, j in enumerate(order):
        running = max(running, min(1.0, (m - step) * arr[j]))
        adj[j] = running
    return adj.tolist()


def main() -> int:
    # Primary.
    pri_au, pri_ap = _per_fold(VARIANT_ROWS[0][1], VARIANT_ROWS[0][2], VARIANT_ROWS[0][3])
    pri_auroc = pri_au.mean(); pri_auprc = pri_ap.mean()
    pri_au_sd = pri_au.std(ddof=1); pri_ap_sd = pri_ap.std(ddof=1)

    raw_rows = []
    for display, tag, d, tmpl in VARIANT_ROWS[1:]:
        au, ap = _per_fold(tag, d, tmpl)
        delta_auroc = (au - pri_au).mean()
        delta_auprc = (ap - pri_ap).mean()
        p_auroc = stats.wilcoxon(au - pri_au, alternative="two-sided", zero_method="wilcox").pvalue
        p_auprc = stats.wilcoxon(ap - pri_ap, alternative="two-sided", zero_method="wilcox").pvalue
        raw_rows.append((display, au.mean(), au.std(ddof=1), ap.mean(), ap.std(ddof=1),
                         delta_auroc, delta_auprc, p_auroc, p_auprc))

    holm_au = _holm([r[7] for r in raw_rows])
    holm_ap = _holm([r[8] for r in raw_rows])

    print(f"{'Variant':<60} {'AUROC':>14} {'AUPRC':>14} "
          f"{'ΔAUROC':>9} {'p_Holm':>9} {'ΔAUPRC':>9} {'p_Holm':>9}")
    print("-" * 132)
    print(f"{'HerbPairIAM (primary)':<60} {pri_auroc:.3f}±{pri_au_sd:.3f}    "
          f"{pri_auprc:.3f}±{pri_ap_sd:.3f}    {'—':>9} {'—':>9} {'—':>9} {'—':>9}")
    for (display, auroc, au_sd, auprc, ap_sd, d_au, d_ap, p_au, p_ap), h_au, h_ap in zip(raw_rows, holm_au, holm_ap):
        print(f"{display:<60} {auroc:.3f}±{au_sd:.3f}    "
              f"{auprc:.3f}±{ap_sd:.3f}    "
              f"{d_au:+.3f} {h_au:9.3g} {d_ap:+.3f} {h_ap:9.3g}")

    # Print LaTeX-ready numbers too, in the exact format of Main Table 3.
    print()
    print("=== LaTeX-ready ===")
    print(f"HerbPairIAM (primary) & ${pri_auroc:.3f}\\pm{pri_au_sd:.3f}$ & "
          f"${pri_auprc:.3f}\\pm{pri_ap_sd:.3f}$ & --- & --- \\\\")
    for (display, auroc, au_sd, auprc, ap_sd, d_au, d_ap, p_au, p_ap), h_au, h_ap in zip(raw_rows, holm_au, holm_ap):
        pau_s = f"\\num{{{h_au:.2g}}}" if h_au < 0.1 else f"${h_au:.2f}$"
        pap_s = f"\\num{{{h_ap:.2g}}}" if h_ap < 0.1 else f"${h_ap:.2f}$"
        sign = "+" if d_au >= 0 else ""
        print(f"\\quad {display} & ${auroc:.3f}\\pm{au_sd:.3f}$ & "
              f"${auprc:.3f}\\pm{ap_sd:.3f}$ & "
              f"${sign}{d_au:.3f}$ [{pau_s}] & "
              f"${'+' if d_ap>=0 else ''}{d_ap:.3f}$ [{pap_s}] \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
