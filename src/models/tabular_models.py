"""Tabular baselines: LogReg, RandomForest, GradientBoosting, MLP, XGBoost."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from experiment_utils import compute_metrics, fold_result_path, save_pickle, val_score_from_metrics


MODEL_GRIDS = {
    "LogisticRegression": [
        {"C": c}
        for c in [0.1, 1.0, 3.0]
    ],
    "RandomForest": [
        {"n_estimators": n, "max_depth": d}
        for n, d in product([200, 500], [5, 10])
    ],
    "XGBoost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss, "colsample_bytree": cs}
        for n, d, lr, ss, cs in product([300, 500], [3, 5], [0.05, 0.1], [1.0], [0.8])
    ],
    "MLP": [
        {"hidden_layer_sizes": h, "alpha": alpha}
        for h, alpha in product([(64, 32), (128, 64)], [1e-4, 1e-3])
    ],
    "GradientBoosting": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n, d, lr in product([200, 500], [3], [0.05, 0.1])
    ],
}


def make_model(model_name: str, params: dict):
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42, **params)
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if model_name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            reg_lambda=1.0,
            min_child_weight=1,
            **params,
        )
    if model_name == "MLP":
        return MLPClassifier(max_iter=500, random_state=42, **params)
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42, **params)
    raise ValueError(f"Unknown tabular model: {model_name}")


def needs_scaler(model_name: str) -> bool:
    return model_name in {"LogisticRegression", "MLP"}


def balance_train_indices(labels, train_idx, neg_ratio=1, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx, dtype=int)
    labels = np.asarray(labels, dtype=int)
    pos_idx = train_idx[labels[train_idx] == 1]
    neg_idx = train_idx[labels[train_idx] == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return train_idx
    n_neg = min(len(neg_idx), len(pos_idx) * int(neg_ratio))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    balanced = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(balanced)
    return balanced


def _slice_features(X, indices, feature_idx=None):
    X_slice = X[indices]
    if feature_idx is not None:
        X_slice = X_slice[:, feature_idx]
    return X_slice


def fit_predict_split(model_name, X, y, split, params, neg_ratio=1, seed=42, feature_idx=None):
    train_idx = np.asarray(split["train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)

    balanced_idx = balance_train_indices(y, train_idx, neg_ratio=neg_ratio, seed=seed + int(split.get("fold", 0)))
    X_train = _slice_features(X, balanced_idx, feature_idx)
    y_train = y[balanced_idx]
    X_val = _slice_features(X, val_idx, feature_idx)
    y_val = y[val_idx]
    X_test = _slice_features(X, test_idx, feature_idx)
    y_test = y[test_idx]

    scaler = None
    if needs_scaler(model_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model = make_model(model_name, params)
    model.fit(X_train, y_train)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_prob_val)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_prob_test, threshold=val_metrics["threshold"])

    result = {
        "model": model_name,
        "fold": int(split.get("fold", split.get("seed", 0))),
        "split_key": "fold" if "fold" in split else "seed",
        "y_true": y_test,
        "y_prob": y_prob_test,
        "test_indices": test_idx,
        "y_true_val": y_val,
        "y_prob_val": y_prob_val,
        "val_indices": val_idx,
        "threshold": float(val_metrics["threshold"]),
        **{k: float(v) for k, v in test_metrics.items()},
        "train_time_sec": 0.0,
        "inference_time_sec": 0.0,
        "peak_memory_mb": float("nan"),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "config": {"params": params, "neg_ratio": neg_ratio, "feature_idx": feature_idx},
    }
    return result


def search_params(model_name, X, y, split, neg_ratio=1, feature_idx=None):
    best_params = MODEL_GRIDS[model_name][0]
    best_score = -np.inf
    for params in MODEL_GRIDS[model_name]:
        result = fit_predict_split(model_name, X, y, split, params, neg_ratio=neg_ratio, seed=42, feature_idx=feature_idx)
        val_metrics = compute_metrics(result["y_true_val"], result["y_prob_val"])
        metric = val_score_from_metrics(val_metrics)
        if metric > best_score:
            best_score = metric
            best_params = params
    return best_params


def run_cv(model_name, X, y, fold_splits, params, neg_ratio=1, feature_idx=None, fold_subset=None):
    results = []
    for split in fold_splits:
        split_id = int(split.get("fold", split.get("seed", 0)))
        if fold_subset is not None and split_id not in fold_subset:
            continue
        result = fit_predict_split(
            model_name=model_name,
            X=X,
            y=y,
            split=split,
            params=params,
            neg_ratio=neg_ratio,
            seed=42,
            feature_idx=feature_idx,
        )
        save_pickle(result, fold_result_path(model_name, split_id))
        results.append(result)
    return results


def summarize_results(results):
    summary = {}
    for key in ["auroc", "auprc", "precision", "recall", "f1", "mcc"]:
        vals = np.asarray([float(r[key]) for r in results], dtype=float)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std(ddof=0))
    return summary
