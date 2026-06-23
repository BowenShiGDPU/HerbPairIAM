"""Compute clinical-utility metrics (DCA, calibration, Brier, ECE) on OOF predictions.

For every model with a complete set of fold pickles under
``EXPERIMENT_SUBDIR/fold_results/`` we concatenate the out-of-fold ``y_true``
and ``y_prob`` and produce:

1. **Decision Curve Analysis (Vickers & Elkin 2006)** — net benefit curve
   across a grid of risk thresholds, plus the ``treat all`` and
   ``treat none`` reference curves. Written to::

       supplementary/decision_curve.csv
       figures/decision_curve.png

2. **Calibration** — reliability diagram (binned mean predicted
   probability vs. observed frequency) and scalar summaries:
   **Brier score** (quadratic loss) and **Expected Calibration Error**
   (ECE, equal-width binning). Written to::

       supplementary/calibration_curves.csv
       supplementary/calibration_summary.csv
       figures/calibration.png

The script is read-only w.r.t. training state (no retraining) and
respects the intrinsic ``model`` key of each fold pickle (not the
filename) so it can be run on any experiment subdirectory that contains
fold pickles with ``y_true`` / ``y_prob``.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/scripts/compute_clinical_metrics.py [--min-folds 10]
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
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import FIGURES_DIR, FOLD_RESULTS_DIR, SUPP_DIR, ensure_output_dirs, load_pickle


sys.stdout.reconfigure(line_buffering=True)

# Matplotlib is imported lazily only when we actually produce figures so that
# running this script on a headless machine without X works.


_FOLD_RE = re.compile(r"^(?P<model>.+)_fold\d+$")


def _load_by_filename_model() -> dict[str, list[dict]]:
    """Return {filename_model_prefix: [fold_result_dict, ...]}.

    Groups by filename (not obj['model']) so that hard-linked / renamed
    fold pickles are attributed to the model whose name they advertise.
    """
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
    for k in out:
        out[k] = sorted(out[k], key=lambda r: int(r.get("fold", 0)))
    return out


def _pool(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y_t = np.concatenate([np.asarray(r["y_true"], dtype=int) for r in results])
    y_p = np.concatenate([np.asarray(r["y_prob"], dtype=float) for r in results])
    return y_t, y_p


# ---------------------------------------------------------------------------
# Decision Curve Analysis
# ---------------------------------------------------------------------------
def decision_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Vickers & Elkin (2006) net benefit.

    net_benefit(p_t) = TP/N - FP/N * (p_t / (1 - p_t))

    Vectorised over the threshold grid. Returns an array of net benefits the
    same length as ``thresholds``. Undefined at p_t=1 (division by zero); we
    clip to p_t < 0.99.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    N = float(len(y_true))
    nb = np.empty_like(thresholds, dtype=float)
    for i, pt in enumerate(thresholds):
        if pt >= 0.999:
            nb[i] = np.nan
            continue
        pred = (y_prob >= pt).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        if pt >= 1.0:
            nb[i] = np.nan
            continue
        nb[i] = tp / N - fp / N * (pt / (1.0 - pt))
    return nb


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """`Treat all` reference: NB = prevalence - (1-prevalence) * p_t/(1-p_t)."""
    prev = float(y_true.mean())
    nb = np.empty_like(thresholds, dtype=float)
    for i, pt in enumerate(thresholds):
        if pt >= 0.999:
            nb[i] = np.nan
            continue
        nb[i] = prev - (1.0 - prev) * (pt / (1.0 - pt))
    return nb


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width binning calibration table.

    Returns one row per bin with ``bin_lower`` / ``bin_upper`` / ``n`` /
    ``mean_predicted`` / ``observed_frequency`` columns. Empty bins are
    still emitted with ``n=0`` so the table is a reproducible grid.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n > 0:
            mp = float(y_prob[mask].mean())
            of = float(y_true[mask].mean())
        else:
            mp = float("nan")
            of = float("nan")
        rows.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "n": n,
            "mean_predicted": mp,
            "observed_frequency": of,
        })
    return pd.DataFrame(rows)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    cal = calibration_table(y_true, y_prob, n_bins=n_bins)
    w = cal["n"].to_numpy(dtype=float) / max(int(cal["n"].sum()), 1)
    err = np.abs(cal["mean_predicted"].to_numpy() - cal["observed_frequency"].to_numpy())
    # ignore empty bins
    mask = (cal["n"].to_numpy() > 0) & np.isfinite(err)
    return float((w[mask] * err[mask]).sum())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_decision_curves(dca_df: pd.DataFrame, outfile: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    models = sorted([m for m in dca_df["model"].unique() if m not in {"treat_all", "treat_none"}])
    for m in models:
        sub = dca_df[dca_df["model"] == m]
        ax.plot(sub["threshold"], sub["net_benefit"], label=m, linewidth=1.5)
    ref_all = dca_df[dca_df["model"] == "treat_all"]
    if not ref_all.empty:
        ax.plot(ref_all["threshold"], ref_all["net_benefit"], label="treat all",
                linestyle="--", color="#999", linewidth=1.2)
    ax.axhline(0.0, linestyle="--", color="#333", linewidth=1.0, label="treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis (pooled OOF)")
    # y-axis often dominated by very-low-threshold region; clip sensibly.
    finite = dca_df[np.isfinite(dca_df["net_benefit"])]
    if not finite.empty:
        lo = float(finite["net_benefit"].quantile(0.02))
        hi = float(finite["net_benefit"].quantile(0.98))
        margin = max((hi - lo) * 0.1, 0.02)
        ax.set_ylim(lo - margin, hi + margin)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def _plot_calibration(cal_frames: list[tuple[str, pd.DataFrame]], outfile: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for model, frame in cal_frames:
        x = frame["mean_predicted"].to_numpy()
        y = frame["observed_frequency"].to_numpy()
        mask = (frame["n"].to_numpy() > 0) & np.isfinite(x) & np.isfinite(y)
        ax.plot(x[mask], y[mask], marker="o", label=model, linewidth=1.2, markersize=5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#333", linewidth=1.0, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (pooled OOF)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-folds", type=int, default=10,
                        help="Minimum fold pickles required to include a model (default 10).")
    parser.add_argument("--threshold-grid", type=float, nargs=3, default=[0.01, 0.95, 0.01],
                        metavar=("START", "STOP", "STEP"),
                        help="Decision-curve threshold grid.")
    parser.add_argument("--n-bins", type=int, default=10, help="Bins for calibration/ECE.")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()

    models = _load_by_filename_model()
    models = {m: rs for m, rs in models.items() if len(rs) >= args.min_folds}
    if not models:
        print(f"No fold pickles with >= {args.min_folds} folds in {FOLD_RESULTS_DIR}")
        return 1

    # Build DCA long-format CSV.
    start, stop, step = args.threshold_grid
    thresholds = np.arange(start, stop + 1e-9, step)
    dca_rows = []
    cal_frames = []
    cal_rows = []
    summary_rows = []
    for m, rs in sorted(models.items()):
        y_t, y_p = _pool(rs)
        nb = decision_curve(y_t, y_p, thresholds)
        for t, v in zip(thresholds, nb):
            dca_rows.append({"model": m, "threshold": float(t), "net_benefit": float(v)})
        cal = calibration_table(y_t, y_p, n_bins=args.n_bins)
        cal.insert(0, "model", m)
        cal_rows.append(cal)
        cal_frames.append((m, cal))
        summary_rows.append({
            "model": m,
            "n_pooled_samples": int(y_t.size),
            "prevalence": float(y_t.mean()),
            "brier": brier_score(y_t, y_p),
            "ece_10bin": expected_calibration_error(y_t, y_p, n_bins=args.n_bins),
        })
    # Reference curves.
    # Any model lets us compute treat_all / treat_none with its own prevalence;
    # since all models share the same OOF indices by construction we take the
    # first listed model as the reference.
    ref_model = sorted(models)[0]
    y_t_ref, _ = _pool(models[ref_model])
    nb_all = treat_all_net_benefit(y_t_ref, thresholds)
    for t, v in zip(thresholds, nb_all):
        dca_rows.append({"model": "treat_all", "threshold": float(t), "net_benefit": float(v)})
    for t in thresholds:
        dca_rows.append({"model": "treat_none", "threshold": float(t), "net_benefit": 0.0})

    dca_df = pd.DataFrame(dca_rows)
    dca_out = SUPP_DIR / "decision_curve.csv"
    dca_df.to_csv(dca_out, index=False)
    print(f"Wrote {dca_out}  rows={len(dca_df)}")

    cal_df = pd.concat(cal_rows, ignore_index=True)
    cal_out = SUPP_DIR / "calibration_curves.csv"
    cal_df.to_csv(cal_out, index=False)
    print(f"Wrote {cal_out}  rows={len(cal_df)}")

    summ_df = pd.DataFrame(summary_rows).sort_values("brier").reset_index(drop=True)
    summ_out = SUPP_DIR / "calibration_summary.csv"
    summ_df.to_csv(summ_out, index=False)
    print(f"Wrote {summ_out}  rows={len(summ_df)}")
    print(summ_df.to_string(index=False))

    if not args.skip_figures:
        try:
            _plot_decision_curves(dca_df, FIGURES_DIR / "decision_curve.png")
            print(f"Wrote {FIGURES_DIR / 'decision_curve.png'}")
        except Exception as exc:
            print(f"[warn] decision-curve figure failed: {exc}")
        try:
            _plot_calibration(cal_frames, FIGURES_DIR / "calibration.png")
            print(f"Wrote {FIGURES_DIR / 'calibration.png'}")
        except Exception as exc:
            print(f"[warn] calibration figure failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
