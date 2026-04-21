import math
import os
import pickle
import random
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
OUT_DIR = ROOT_DIR / "outputs"
RESULTS_ROOT_DIR = Path(os.environ.get("RESULTS_ROOT_DIR", str(OUT_DIR)))
if not RESULTS_ROOT_DIR.is_absolute():
    RESULTS_ROOT_DIR = ROOT_DIR / RESULTS_ROOT_DIR
EXPERIMENT_SUBDIR = os.environ.get("EXPERIMENT_SUBDIR", "full_experiment")
# Early stopping, grid search on a single fold, bake-off ranking, graph config search.
# Set to "auprc" to reproduce legacy behavior. Default "auroc" matches primary reporting metric.
VAL_SELECTION_METRIC = os.environ.get("VAL_SELECTION_METRIC", "auroc").strip().lower()
if VAL_SELECTION_METRIC not in ("auroc", "auprc"):
    raise ValueError(f"VAL_SELECTION_METRIC must be 'auroc' or 'auprc', got {VAL_SELECTION_METRIC!r}")


def val_score_from_metrics(metrics: dict) -> float:
    """Score used for model selection / early stopping on the validation set."""
    return float(metrics[VAL_SELECTION_METRIC])


FULL_EXPERIMENT_DIR = RESULTS_ROOT_DIR / EXPERIMENT_SUBDIR
FOLD_RESULTS_DIR = FULL_EXPERIMENT_DIR / "fold_results"
TABLES_DIR = FULL_EXPERIMENT_DIR / "tables"
SUPP_DIR = FULL_EXPERIMENT_DIR / "supplementary"
INTERPRET_DIR = FULL_EXPERIMENT_DIR / "interpretability"
FIGURES_DIR = FULL_EXPERIMENT_DIR / "figures"
MODELS_DIR = FULL_EXPERIMENT_DIR / "models"


def ensure_output_dirs():
    for path in [
        OUT_DIR,
        RESULTS_ROOT_DIR,
        FULL_EXPERIMENT_DIR,
        FOLD_RESULTS_DIR,
        TABLES_DIR,
        SUPP_DIR,
        INTERPRET_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Run manifest
# ----------------------------------------------------------------------

def _git_commit_hash() -> str:
    """Return the current git HEAD commit hash, or '' if unavailable."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _git_is_dirty() -> bool:
    """Return True if there are uncommitted changes (staged or unstaged)."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "status", "--porcelain"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return False


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file, or '' if not readable. Used for dataset.pkl."""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _package_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return ""


def write_run_manifest(
    experiment_subdir: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a ``run_manifest.json`` under ``experiment_subdir`` (or the
    current experiment directory) with git commit, dataset SHA-256,
    environment variables, Python/package versions, and host info.
    """
    import datetime as _dt
    import json as _json
    import socket
    import sys as _sys

    if experiment_subdir is None:
        target = FULL_EXPERIMENT_DIR
    else:
        target = RESULTS_ROOT_DIR / experiment_subdir
    target.mkdir(parents=True, exist_ok=True)

    dataset_path = OUT_DIR / "dataset.pkl"
    manifest = {
        "schema_version": 1,
        "written_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "experiment_subdir": str(target.relative_to(ROOT_DIR) if target.is_relative_to(ROOT_DIR) else target),
        "git": {
            "commit": _git_commit_hash(),
            "dirty": _git_is_dirty(),
        },
        "dataset": {
            "path": str(dataset_path.relative_to(ROOT_DIR) if dataset_path.is_relative_to(ROOT_DIR) else dataset_path),
            "sha256": _file_sha256(dataset_path) if dataset_path.exists() else "",
            "exists": dataset_path.exists(),
        },
        "env_vars": {
            "VAL_SELECTION_METRIC": os.environ.get("VAL_SELECTION_METRIC", ""),
            "RESULTS_ROOT_DIR": os.environ.get("RESULTS_ROOT_DIR", ""),
            "EXPERIMENT_SUBDIR": os.environ.get("EXPERIMENT_SUBDIR", ""),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        },
        "python": {
            "version": _sys.version,
            "argv": list(_sys.argv),
        },
        "packages": {
            name: _package_version(name) for name in [
                "numpy", "scipy", "pandas", "scikit-learn",
                "torch", "torch-geometric", "xgboost",
                "matplotlib", "networkx", "tqdm",
            ]
        },
        "host": {
            "name": socket.gethostname(),
        },
    }
    # CUDA / GPU info (best-effort).
    try:
        import torch as _torch
        manifest["host"]["cuda_available"] = bool(_torch.cuda.is_available())
        if _torch.cuda.is_available():
            manifest["host"]["cuda_version"] = _torch.version.cuda or ""
            manifest["host"]["n_gpus"] = int(_torch.cuda.device_count())
            manifest["host"]["gpu_0_name"] = _torch.cuda.get_device_name(0)
    except Exception:
        pass

    if extra:
        manifest["extra"] = extra

    manifest_path = target / "run_manifest.json"
    # If a manifest already exists, append to a rotating history so that
    # repeated runs against the same subdir don't silently overwrite the
    # provenance of earlier fold pickles.
    history_path = target / "run_manifest_history.jsonl"
    with open(manifest_path, "w") as fh:
        _json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    with open(history_path, "a") as fh:
        fh.write(_json.dumps(manifest, sort_keys=True, default=str) + "\n")
    return manifest_path


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("_") or "model"


def fold_result_path(model_name: str, fold_id: int) -> Path:
    return FOLD_RESULTS_DIR / f"{sanitize_name(model_name)}_fold{fold_id}.pkl"


def model_state_path(model_name: str, fold_id: int) -> Path:
    return MODELS_DIR / f"{sanitize_name(model_name)}_fold{fold_id}.pt"


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def set_seed(seed: int):
    """Seed Python, NumPy and (if available) PyTorch with the same integer."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass


def enable_strict_determinism() -> None:
    """Opt-in strict determinism, gated by ``STRICT_DETERMINISM=1``.

    When enabled, sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and turns on
    ``torch.use_deterministic_algorithms(True, warn_only=True)``. Default
    is a no-op because strict mode slows CUDA training 2-3x.
    """
    if os.environ.get("STRICT_DETERMINISM", "") not in {"1", "true", "True"}:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def select_threshold_max_f1(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(candidates) > 200:
        candidates = np.linspace(0.0, 1.0, 201)

    best_threshold = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if score > best_f1 or (math.isclose(score, best_f1) and recall > best_recall):
            best_f1 = score
            best_recall = recall
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(y_true, y_prob, threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "threshold": 0.5 if threshold is None else float(threshold),
        }

    if len(np.unique(y_true)) < 2:
        auroc = 0.5
        auprc = float(y_true.mean())
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    if threshold is None:
        threshold = select_threshold_max_f1(y_true, y_prob)
    threshold = float(threshold)
    y_pred = (y_prob >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0
    if np.isnan(mcc):
        mcc = 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "threshold": threshold,
    }


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n_tests = len(ordered)
    adjusted = {}
    running_max = 0.0
    for rank, (name, pvalue) in enumerate(ordered):
        corrected = (n_tests - rank) * pvalue
        corrected = min(1.0, corrected)
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted


def paired_ttest_matrix(model_to_fold_results: dict[str, list[dict]], metric: str) -> pd.DataFrame:
    models = list(model_to_fold_results)
    matrix = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    for i, lhs in enumerate(models):
        lhs_vals = [float(r[metric]) for r in model_to_fold_results[lhs]]
        for rhs in models[i + 1 :]:
            rhs_vals = [float(r[metric]) for r in model_to_fold_results[rhs]]
            _, pvalue = stats.ttest_rel(lhs_vals, rhs_vals)
            matrix.loc[lhs, rhs] = float(pvalue)
            matrix.loc[rhs, lhs] = float(pvalue)
        matrix.loc[lhs, lhs] = 0.0
    return matrix


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for bin_id in range(n_bins):
        lo, hi = bins[bin_id], bins[bin_id + 1]
        if bin_id == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            rows.append(
                {
                    "bin": bin_id,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": 0,
                    "mean_pred": np.nan,
                    "empirical_pos_rate": np.nan,
                }
            )
            continue
        mean_pred = float(y_prob[mask].mean())
        empirical = float(y_true[mask].mean())
        weight = mask.mean()
        ece += abs(mean_pred - empirical) * weight
        rows.append(
            {
                "bin": bin_id,
                "bin_lo": lo,
                "bin_hi": hi,
                "count": int(mask.sum()),
                "mean_pred": mean_pred,
                "empirical_pos_rate": empirical,
            }
        )
    table = pd.DataFrame(rows)
    table["ece"] = ece
    return table


def bootstrap_ci(
    y_true,
    y_prob,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    stats_out = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        sample_prob = y_prob[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        stats_out.append(metric_fn(sample_true, sample_prob))
    if not stats_out:
        return (np.nan, np.nan)
    lo, hi = np.percentile(stats_out, [2.5, 97.5])
    return float(lo), float(hi)


def compute_peak_memory_mb() -> float:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024**2))
    except Exception:
        return float("nan")


def time_call(fn: Callable, *args, **kwargs):
    start = time.time()
    output = fn(*args, **kwargs)
    elapsed = time.time() - start
    return output, elapsed


def compute_pooled_predictions(fold_results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.asarray(fr["y_true"], dtype=int) for fr in fold_results])
    y_prob = np.concatenate([np.asarray(fr["y_prob"], dtype=float) for fr in fold_results])
    return y_true, y_prob


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1) + 1.0
        midranks[i:j] = midrank
        i = j
    result = np.empty(n, dtype=float)
    result[order] = midranks
    return result


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_test(y_true, y_prob_a, y_prob_b) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob_a = np.asarray(y_prob_a, dtype=float)
    y_prob_b = np.asarray(y_prob_b, dtype=float)
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    preds = np.vstack([y_prob_a[order], y_prob_b[order]])
    label_1_count = int(y_true_sorted.sum())
    if label_1_count == 0 or label_1_count == len(y_true_sorted):
        return float("nan")
    aucs, covariance = _fast_delong(preds, label_1_count)
    diff = np.abs(aucs[0] - aucs[1])
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    variance = max(float(variance), 1e-12)
    z = diff / math.sqrt(variance)
    return float(2 * stats.norm.sf(z))
