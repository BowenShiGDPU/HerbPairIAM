"""Phase 5 interpretability: leakage-safe artefacts mandated by EXPERIMENT_PROTOCOL.md s10.

Outputs (all written under the active stage's ``interpretability/`` directory):

* ``case_{01..15}.json`` + ``case_summary.csv`` -- s10.1
* ``pmda_concordance.csv``                     -- s10.2
* ``herb_attention_consistency.csv``           -- s10.3
* ``top_novel_predictions.csv``                -- s10.4 (label==0 high-prob set)

Only existing fold artefacts are read. We rebuild the DoseAwareIAM forward pass
with the saved ``models/DoseAwareIAM_fold*.pt`` weights to recover per-sample
herb / pair attention. Each test sample is attributed to exactly one fold
(its OOF fold) so the analysis is held-out by construction.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/main_benchmark \\
    python -u src/phase5_interpretability.py [--mode {all,cases,pmda,attn,novel}]

The PMDA known-interaction list lives in ``data/pmda_known_interactions.yaml``
when present and falls back to a small built-in seed otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from experiment_utils import (
    FIGURES_DIR,
    FOLD_RESULTS_DIR,
    INTERPRET_DIR,
    MODELS_DIR,
    SUPP_DIR,
    compute_metrics,
    ensure_output_dirs,
    load_pickle,
    save_pickle,
)
from neural_models import (
    ModelConfig,
    build_model as build_runtime_model,
    build_sample_collections,
    load_all,
)


sys.stdout.reconfigure(line_buffering=True)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "final_data_clean"
OUT_DIR = ROOT_DIR / "outputs"
PMDA_DEFAULT_PATH = ROOT_DIR / "data" / "pmda_known_interactions.yaml"
DEFAULT_FROZEN_CONFIG = ModelConfig(
    hidden=32, dropout=0.3, lr=1e-3, epochs=100, patience=10, neg_ratio=10, eval_every=2
)
DEFAULT_TOP_K_NOVEL = 20
DEFAULT_N_CASES = 15


PMDA_FALLBACK_INTERACTIONS = [
    {
        "name": "Glycyrrhiza-induced hypokalaemia",
        "herb_id": "CMM20",
        "herb_name_en_hint": "Glycyrrhiza",
        "adr_name_substrings": ["HYPOKALAEMIA", "HYPOKALEMIA"],
        "mechanism": "Glycyrrhizic acid inhibits 11-beta-HSD2, increasing renal K+ excretion.",
        "source": "PMDA Section 10 / pharmacology",
    },
    {
        "name": "Glycyrrhiza-induced pseudoaldosteronism",
        "herb_id": "CMM20",
        "herb_name_en_hint": "Glycyrrhiza",
        "adr_name_substrings": ["PSEUDOALDOSTERONISM"],
        "mechanism": "Apparent mineralocorticoid excess via 11-beta-HSD2 inhibition.",
        "source": "PMDA Section 10",
    },
    {
        "name": "Scutellaria baicalensis interstitial pneumonia",
        "herb_id": "CMM08",
        "herb_name_en_hint": "Scutellaria Root",
        "adr_name_substrings": ["INTERSTITIAL LUNG DISEASE", "INTERSTITIAL PNEUMON"],
        "mechanism": "Reported lymphocytic alveolitis associated with Scutellaria baicalensis-containing kampo formulas.",
        "source": "PMDA Section 10",
    },
    {
        "name": "Ephedra-induced cardiovascular events",
        "herb_id": "CMM113",
        "herb_name_en_hint": "Ephedra Herb",
        "adr_name_substrings": ["PALPITATIONS", "TACHYCARDIA", "HYPERTENSION", "ARRHYTHMI"],
        "mechanism": "Ephedrine sympathomimetic effect on alpha/beta adrenergic receptors.",
        "source": "PMDA Section 10",
    },
    {
        "name": "Aconite-induced cardiac arrhythmia",
        "herb_id": "CMM106",
        "herb_name_en_hint": "Processed Aconite Root",
        "adr_name_substrings": ["ARRHYTHMI", "TACHYCARDIA", "PALPITATIONS"],
        "mechanism": "Aconitine is a Na+ channel agonist on cardiomyocytes.",
        "source": "PMDA Section 10",
    },
    {
        "name": "Glycyrrhiza-related rhabdomyolysis",
        "herb_id": "CMM20",
        "herb_name_en_hint": "Glycyrrhiza",
        "adr_name_substrings": ["RHABDOMYOLYSIS"],
        "mechanism": "Hypokalaemia secondary to chronic glycyrrhizin exposure leading to muscle injury.",
        "source": "PMDA Section 10",
    },
    {
        "name": "Bupleurum-induced hepatic injury",
        "herb_id": "CMM42",
        "herb_name_en_hint": "Bupleurum Root",
        "adr_name_substrings": ["HEPATIC FUNCTION ABNORMAL", "LIVER INJURY", "HEPATITIS"],
        "mechanism": "Idiosyncratic hepatotoxicity reported with Sho-saiko-to and other Bupleurum-containing formulas.",
        "source": "PMDA Section 10",
    },
]


def _load_yaml_if_present(path: Path):
    if not path.exists():
        return None
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        warnings.warn(f"Could not parse PMDA list at {path}: {exc}")
        return None


def _resolve_pmda_interactions(custom_yaml: Path | None) -> List[dict]:
    target = custom_yaml or PMDA_DEFAULT_PATH
    user = _load_yaml_if_present(target)
    if user:
        return list(user)
    return list(PMDA_FALLBACK_INTERACTIONS)


def _load_node_metadata():
    cmm = pd.read_csv(DATA_DIR / "CMM_nodes.csv")
    adr = pd.read_csv(DATA_DIR / "ADR_nodes.csv")
    ing = pd.read_csv(DATA_DIR / "Ingredient_nodes.csv")
    formula = pd.read_csv(DATA_DIR / "TCMF_nodes.csv")
    herb_name = dict(zip(cmm["CMM_id"], cmm["herb_name_en"]))
    adr_name = dict(zip(adr["Adr_id"], adr["ADR_name"]))
    formula_name = {row["TCMF_id"]: row.get("formula_name_jp", row["TCMF_id"]) for _, row in formula.iterrows()}
    ing_name = {}
    for _, row in ing.iterrows():
        candidate = row.get("ingredient_name") or row.get("molecule_name") or row.get("In_id")
        ing_name[row["In_id"]] = candidate if pd.notna(candidate) else row["In_id"]
    return herb_name, adr_name, formula_name, ing_name


def _load_pathway_names():
    pw_df = pd.read_csv(DATA_DIR / "multiomics" / "kg_edges_ta_reactome.tsv", sep="\t")
    return dict(zip(pw_df["reactome_id"], pw_df["pathway_name"]))


def _build_models_for_inference(model_name: str, sample_example, config_dict: dict | None = None):
    cfg_template = {**DEFAULT_FROZEN_CONFIG.__dict__, **(config_dict or {})}
    cfg = ModelConfig(**{k: v for k, v in cfg_template.items() if k in ModelConfig().__dict__})
    return build_runtime_model(model_name, cfg, sample_example=sample_example)


@torch.no_grad()
def collect_oof_attention_predictions(model_name: str, ds, samples, config_dict: dict | None = None) -> pd.DataFrame:
    sample_example = next((s for s in samples if s is not None), None)
    rows = []
    for split in ds["fold_splits"]:
        fold_id = int(split["fold"])
        state_path = MODELS_DIR / f"{model_name}_fold{fold_id}.pt"
        if not state_path.exists():
            print(f"  WARN: missing weights {state_path}, fold {fold_id} skipped.")
            continue
        model = _build_models_for_inference(model_name, sample_example, config_dict=config_dict)
        try:
            state = torch.load(state_path, map_location="cpu")
        except Exception as exc:
            print(f"  WARN: cannot load {state_path}: {exc}")
            continue
        load_info = model.load_state_dict(state, strict=False)
        if load_info.missing_keys or load_info.unexpected_keys:
            print(
                f"  NOTE: fold {fold_id} weights loaded with non-strict matching; "
                f"missing={list(load_info.missing_keys)[:3]}{'...' if len(load_info.missing_keys) > 3 else ''} "
                f"unexpected={list(load_info.unexpected_keys)[:3]}{'...' if len(load_info.unexpected_keys) > 3 else ''}",
                flush=True,
            )
        model.eval()
        for idx in split["test_idx"]:
            sample = samples[int(idx)]
            if sample is None:
                continue
            logit, aux = model(sample)
            herb_attn = aux.get("herb_attn")
            pair_attn = aux.get("pair_attn")
            rows.append(
                {
                    "sample_idx": int(idx),
                    "fold": fold_id,
                    "prob": float(torch.sigmoid(logit).item()),
                    "herb_attn": (herb_attn.detach().cpu().numpy() if herb_attn is not None else np.asarray([])),
                    "pair_attn": (pair_attn.detach().cpu().numpy() if pair_attn is not None else np.asarray([])),
                }
            )
    return pd.DataFrame(rows)


def _herb_pair_score(pf: dict | None, dose_product: float = 0.0) -> float:
    if pf is None:
        return float("-inf")
    convergence = float(pf.get("convergence", 0.0))
    complementarity = float(pf.get("complementarity", 0.0))
    ppi = float(pf.get("ppi_bridge", 0.0))
    return float(convergence + 0.5 * complementarity + 0.1 * min(ppi, 10.0)) + 0.05 * dose_product


def _row_to_attn_array(value, expected_len: int) -> np.ndarray:
    if value is None:
        return np.zeros(expected_len, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != expected_len:
        return np.zeros(expected_len, dtype=float)
    return arr


def _build_case_payload(
    pred_row: dict,
    f2h: dict,
    dose_map: dict,
    herb_name: dict,
    adr_name: dict,
    formula_name: dict,
    ing_name: dict,
    pair_features: dict,
    ingredient_index: dict,
    pathway_t2pw: dict,
    pw_id2name: dict,
) -> dict:
    f_id = pred_row["TCMF_id"]
    a_id = pred_row["Adr_id"]
    herbs = sorted(f2h.get(f_id, []))
    alpha = _row_to_attn_array(pred_row.get("herb_attn"), len(herbs))

    herb_attention = []
    for i, h in enumerate(herbs):
        dr = float(dose_map.get((f_id, h), {}).get("dose_ratio", 0.0))
        herb_attention.append({
            "herb_id": h,
            "herb_name": herb_name.get(h, h),
            "attention": float(alpha[i]) if i < alpha.size else 0.0,
            "dose_ratio": dr,
        })
    herb_attention.sort(key=lambda x: x["attention"], reverse=True)

    pair_records = []
    for h1, h2 in combinations(herbs, 2):
        pf = pair_features.get((h1, h2, a_id))
        if pf is None:
            continue
        dr1 = float(dose_map.get((f_id, h1), {}).get("dose_ratio", 0.0))
        dr2 = float(dose_map.get((f_id, h2), {}).get("dose_ratio", 0.0))
        score = _herb_pair_score(pf, dr1 * dr2)
        pair_records.append({
            "h1_id": h1,
            "h2_id": h2,
            "h1_name": herb_name.get(h1, h1),
            "h2_name": herb_name.get(h2, h2),
            "convergence": float(pf.get("convergence", 0.0)),
            "complementarity": float(pf.get("complementarity", 0.0)),
            "union_coverage": float(pf.get("union_coverage", 0.0)),
            "ppi_bridge": float(pf.get("ppi_bridge", 0.0)),
            "pw_convergence": float(pf.get("pw_convergence", 0.0)),
            "tissue_coexpr": float(pf.get("tissue_coexpr", 0.0)),
            "dose_product": dr1 * dr2,
            "interaction_score": float(score),
        })
    pair_records.sort(key=lambda x: x["interaction_score"], reverse=True)
    top_pairs = pair_records[:5]
    for pair in top_pairs:
        h1, h2 = pair["h1_id"], pair["h2_id"]
        idx = ingredient_index.get((h1, h2, a_id), [])
        ingredient_traceback = []
        for entry in sorted(idx, key=lambda x: len(x[2]), reverse=True)[:3]:
            i1, i2, shared_targets = entry[0], entry[1], entry[2]
            shared_pw = set()
            for t in shared_targets:
                shared_pw |= {pw_id2name.get(p, p) for p in pathway_t2pw.get(t, set())}
            ingredient_traceback.append({
                "i1_id": i1,
                "i2_id": i2,
                "i1_name": ing_name.get(i1, i1),
                "i2_name": ing_name.get(i2, i2),
                "n_shared_adr_targets": int(len(shared_targets)),
                "shared_pathways_top": sorted(list(shared_pw))[:5],
            })
        pair["ingredient_traceback"] = ingredient_traceback

    payload = {
        "TCMF_id": f_id,
        "formula_name": formula_name.get(f_id, f_id),
        "Adr_id": a_id,
        "ADR_name": adr_name.get(a_id, a_id),
        "fold": int(pred_row["fold"]),
        "sample_idx": int(pred_row["sample_idx"]),
        "label": int(pred_row["label"]),
        "predicted_probability": float(pred_row["prob"]),
        "n_herbs": len(herbs),
        "herb_attention_ranked": herb_attention,
        "top_pair_interactions": top_pairs,
    }
    return payload


def _resolve_pair_features() -> Tuple[dict, dict, dict]:
    pair_features = load_pickle(OUT_DIR / "herb_pair_features.pkl")
    ingredient_index = load_pickle(OUT_DIR / "ingredient_pair_index.pkl")
    pw_maps = load_pickle(OUT_DIR / "pathway_tissue_maps.pkl")
    return pair_features, ingredient_index, pw_maps["t2pw"]


def run_cases(
    n_cases: int,
    ds,
    df: pd.DataFrame,
    samples,
    f2h: dict,
    dose_map: dict,
    herb_name: dict,
    adr_name: dict,
    formula_name: dict,
    ing_name: dict,
    pair_features: dict,
    ingredient_index: dict,
    pathway_t2pw: dict,
    pw_id2name: dict,
    oof_predictions: pd.DataFrame,
    out_dir: Path,
):
    if oof_predictions.empty:
        print("  WARN: no OOF predictions available, skipping case JSONs.")
        return pd.DataFrame()
    df_local = df.reset_index(drop=True).copy()
    df_local["sample_idx"] = df_local.index.astype(int)
    merged = df_local.merge(oof_predictions, on="sample_idx", how="inner")
    merged["label"] = merged["label"].astype(int)
    merged = merged.sort_values("prob", ascending=False)
    pos_top = merged[merged["label"] == 1].head(n_cases).to_dict("records")
    summary_rows = []
    for rank, row in enumerate(pos_top, start=1):
        payload = _build_case_payload(
            row, f2h, dose_map, herb_name, adr_name, formula_name, ing_name,
            pair_features, ingredient_index, pathway_t2pw, pw_id2name,
        )
        payload["case_rank"] = rank
        case_path = out_dir / f"case_{rank:02d}.json"
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        summary_rows.append({
            "case_rank": rank,
            "TCMF_id": payload["TCMF_id"],
            "formula_name": payload["formula_name"],
            "Adr_id": payload["Adr_id"],
            "ADR_name": payload["ADR_name"],
            "fold": payload["fold"],
            "label": payload["label"],
            "prob": payload["predicted_probability"],
            "top_herb": payload["herb_attention_ranked"][0]["herb_name"] if payload["herb_attention_ranked"] else "",
            "top_herb_attn": payload["herb_attention_ranked"][0]["attention"] if payload["herb_attention_ranked"] else float("nan"),
            "top_pair": (
                f"{payload['top_pair_interactions'][0]['h1_name']} x {payload['top_pair_interactions'][0]['h2_name']}"
                if payload["top_pair_interactions"]
                else ""
            ),
            "top_pair_score": payload["top_pair_interactions"][0]["interaction_score"] if payload["top_pair_interactions"] else float("nan"),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "case_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  cases -> {len(summary_rows)} JSON files + {summary_path}")
    return summary_df


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj))
    raise TypeError(f"Cannot serialise {type(obj)}")


def run_attention_consistency(
    df: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    f2h: dict,
    herb_name: dict,
    adr_name: dict,
    out_csv: Path,
    min_formulas: int = 2,
):
    if oof_predictions.empty:
        print("  WARN: no OOF predictions; skipping attention consistency.")
        return pd.DataFrame()
    df_local = df.reset_index(drop=True).copy()
    df_local["sample_idx"] = df_local.index.astype(int)
    merged = df_local.merge(oof_predictions, on="sample_idx", how="inner")
    rows = []
    for sample_idx, fold_id, herb_attn, f_id, a_id, prob in zip(
        merged["sample_idx"],
        merged["fold"],
        merged["herb_attn"],
        merged["TCMF_id"],
        merged["Adr_id"],
        merged["prob"],
    ):
        herbs = sorted(f2h.get(f_id, []))
        attn_arr = _row_to_attn_array(herb_attn, len(herbs))
        if attn_arr.size == 0 or float(attn_arr.sum()) == 0.0:
            continue
        ranks = pd.Series(-attn_arr).rank(method="min").astype(int).values
        for h, attn, rank in zip(herbs, attn_arr, ranks):
            rows.append({
                "herb_id": h,
                "Adr_id": a_id,
                "TCMF_id": f_id,
                "fold": int(fold_id),
                "attention": float(attn),
                "rank": int(rank),
                "prob": float(prob),
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print("  WARN: empty attention matrix.")
        return long_df
    grouped = long_df.groupby(["herb_id", "Adr_id"])
    agg_rows = []
    from scipy import stats

    for (herb_id, adr_id), grp in grouped:
        if len(grp) < min_formulas:
            continue
        attn = grp["attention"].to_numpy(dtype=float)
        rank = grp["rank"].to_numpy(dtype=float)
        prob = grp["prob"].to_numpy(dtype=float)
        spearman_attn_prob = float("nan")
        if len(grp) >= 3 and float(np.nanstd(attn)) > 0 and float(np.nanstd(prob)) > 0:
            try:
                spearman_attn_prob = float(stats.spearmanr(attn, prob).correlation)
            except Exception:
                spearman_attn_prob = float("nan")
        agg_rows.append({
            "herb_id": herb_id,
            "herb_name": herb_name.get(herb_id, herb_id),
            "Adr_id": adr_id,
            "ADR_name": adr_name.get(adr_id, adr_id),
            "n_formulas": int(len(grp)),
            "mean_attention": float(np.mean(attn)),
            "std_attention": float(np.std(attn, ddof=0)),
            "mean_rank": float(np.mean(rank)),
            "std_rank": float(np.std(rank, ddof=0)),
            "spearman_attn_vs_prob": spearman_attn_prob,
        })
    out_df = pd.DataFrame(agg_rows).sort_values(["n_formulas", "mean_attention"], ascending=[False, False]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  attention_consistency -> {out_csv} ({len(out_df)} (herb, ADR) pairs)")
    return out_df


def run_top_novel_predictions(
    df: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    f2h: dict,
    herb_name: dict,
    adr_name: dict,
    formula_name: dict,
    pair_features: dict,
    dose_map: dict,
    out_csv: Path,
    top_k: int,
):
    if oof_predictions.empty:
        print("  WARN: no OOF predictions; skipping novel predictions.")
        return pd.DataFrame()
    df_local = df.reset_index(drop=True).copy()
    df_local["sample_idx"] = df_local.index.astype(int)
    merged = df_local.merge(oof_predictions, on="sample_idx", how="inner")
    merged["label"] = merged["label"].astype(int)
    novel = merged[merged["label"] == 0].sort_values("prob", ascending=False).head(top_k).to_dict("records")
    rows = []
    for entry in novel:
        f_id = entry["TCMF_id"]
        a_id = entry["Adr_id"]
        herbs = sorted(f2h.get(f_id, []))
        alpha = _row_to_attn_array(entry.get("herb_attn"), len(herbs))
        top_herb_idx = int(np.argmax(alpha)) if alpha.size > 0 else -1
        top_herb_id = herbs[top_herb_idx] if 0 <= top_herb_idx < len(herbs) else ""
        top_pair_label = ""
        top_pair_score = float("nan")
        best_score = float("-inf")
        for h1, h2 in combinations(herbs, 2):
            pf = pair_features.get((h1, h2, a_id))
            if pf is None:
                continue
            dr1 = float(dose_map.get((f_id, h1), {}).get("dose_ratio", 0.0))
            dr2 = float(dose_map.get((f_id, h2), {}).get("dose_ratio", 0.0))
            sc = _herb_pair_score(pf, dr1 * dr2)
            if sc > best_score:
                best_score = sc
                top_pair_label = f"{herb_name.get(h1, h1)} x {herb_name.get(h2, h2)}"
                top_pair_score = float(sc)
        rows.append({
            "TCMF_id": f_id,
            "formula_name": formula_name.get(f_id, f_id),
            "Adr_id": a_id,
            "ADR_name": adr_name.get(a_id, a_id),
            "fold": int(entry["fold"]),
            "prob": float(entry["prob"]),
            "label_in_dataset": int(entry["label"]),
            "tested_source": str(entry.get("tested_source", "")),
            "top_herb": herb_name.get(top_herb_id, top_herb_id),
            "top_herb_attn": float(alpha[top_herb_idx]) if 0 <= top_herb_idx < alpha.size else float("nan"),
            "top_pair": top_pair_label,
            "top_pair_score": top_pair_score,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"  top_novel_predictions -> {out_csv} ({len(out_df)} rows)")
    return out_df


def run_pmda_concordance(
    interactions: List[dict],
    df: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    f2h: dict,
    herb_name: dict,
    adr_name: dict,
    formula_name: dict,
    out_csv: Path,
    k_values: Iterable[int] = (5, 10, 20),
):
    if oof_predictions.empty:
        print("  WARN: no OOF predictions; skipping PMDA concordance.")
        return pd.DataFrame()
    df_local = df.reset_index(drop=True).copy()
    df_local["sample_idx"] = df_local.index.astype(int)
    merged = df_local.merge(oof_predictions, on="sample_idx", how="inner")
    merged["label"] = merged["label"].astype(int)

    adr_name_to_id: Dict[str, str] = {}
    for adr_id, name in adr_name.items():
        if isinstance(name, str):
            adr_name_to_id.setdefault(name.upper(), adr_id)

    rows = []
    for entry in interactions:
        herb_id = entry.get("herb_id")
        substrings = [s.upper() for s in entry.get("adr_name_substrings", [])]
        candidate_adr_ids = []
        for adr_id, adr_full in adr_name.items():
            if not isinstance(adr_full, str):
                continue
            up = adr_full.upper()
            if any(sub in up for sub in substrings):
                candidate_adr_ids.append(adr_id)
        if not candidate_adr_ids:
            rows.append({
                "interaction": entry.get("name", ""),
                "herb_id": herb_id,
                "herb_name": herb_name.get(herb_id, herb_id),
                "matched_n_adr": 0,
                "n_formulas_containing_herb": int(sum(1 for f, hs in f2h.items() if herb_id in hs)),
                "n_predictions": 0,
                "mean_prob": float("nan"),
                "max_prob": float("nan"),
                "median_rank": float("nan"),
                "best_top_k_hit": "",
                **{f"hit_at_{k}": int(0) for k in k_values},
                **{f"precision_at_{k}": float("nan") for k in k_values},
                "source": entry.get("source", ""),
                "mechanism": entry.get("mechanism", ""),
            })
            continue
        sub = merged[merged["TCMF_id"].apply(lambda f: herb_id in f2h.get(f, [])) & merged["Adr_id"].isin(candidate_adr_ids)]
        if sub.empty:
            rows.append({
                "interaction": entry.get("name", ""),
                "herb_id": herb_id,
                "herb_name": herb_name.get(herb_id, herb_id),
                "matched_n_adr": int(len(candidate_adr_ids)),
                "n_formulas_containing_herb": int(sum(1 for f, hs in f2h.items() if herb_id in hs)),
                "n_predictions": 0,
                "mean_prob": float("nan"),
                "max_prob": float("nan"),
                "median_rank": float("nan"),
                "best_top_k_hit": "",
                **{f"hit_at_{k}": int(0) for k in k_values},
                **{f"precision_at_{k}": float("nan") for k in k_values},
                "source": entry.get("source", ""),
                "mechanism": entry.get("mechanism", ""),
            })
            continue

        merged_sorted = merged.sort_values("prob", ascending=False).reset_index(drop=True)
        merged_sorted["global_rank"] = np.arange(1, len(merged_sorted) + 1, dtype=int)
        sub_with_rank = merged_sorted[
            merged_sorted["TCMF_id"].apply(lambda f: herb_id in f2h.get(f, []))
            & merged_sorted["Adr_id"].isin(candidate_adr_ids)
        ]
        ranks = sub_with_rank["global_rank"].to_numpy(dtype=int)
        probs = sub_with_rank["prob"].to_numpy(dtype=float)
        labels_arr = sub_with_rank["label"].to_numpy(dtype=int)
        row = {
            "interaction": entry.get("name", ""),
            "herb_id": herb_id,
            "herb_name": herb_name.get(herb_id, herb_id),
            "matched_n_adr": int(len(candidate_adr_ids)),
            "n_formulas_containing_herb": int(sum(1 for f, hs in f2h.items() if herb_id in hs)),
            "n_predictions": int(len(sub_with_rank)),
            "n_positives_in_dataset": int(labels_arr.sum()),
            "mean_prob": float(np.mean(probs)) if probs.size else float("nan"),
            "max_prob": float(np.max(probs)) if probs.size else float("nan"),
            "median_rank": float(np.median(ranks)) if ranks.size else float("nan"),
            "best_top_k_hit": "",
            "source": entry.get("source", ""),
            "mechanism": entry.get("mechanism", ""),
        }
        for k in k_values:
            top_k_set = set(merged_sorted.head(k)["sample_idx"])
            hits = int(sub_with_rank["sample_idx"].isin(top_k_set).sum())
            row[f"hit_at_{k}"] = hits
            row[f"precision_at_{k}"] = float(hits / max(k, 1))
            if hits > 0 and not row["best_top_k_hit"]:
                row["best_top_k_hit"] = f"top-{k}"
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"  pmda_concordance -> {out_csv} ({len(out_df)} interactions)")
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=["all", "cases", "pmda", "attn", "novel"])
    parser.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES)
    parser.add_argument("--top-k-novel", type=int, default=DEFAULT_TOP_K_NOVEL)
    parser.add_argument(
        "--pmda-yaml",
        default="",
        help="Optional override path to a PMDA known-interaction YAML; defaults to data/pmda_known_interactions.yaml",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    print(f"Phase 5 interpretability: target stage = {INTERPRET_DIR}")
    if not (FOLD_RESULTS_DIR.exists()):
        print(f"  ERROR: stage directory not initialised: {FOLD_RESULTS_DIR}")
        return 1
    ds, hp, ap, pf, lookups = load_all()
    df = ds["df"].reset_index(drop=True)
    # Prefer the primary model (HerbPairIAM) but fall back to DoseAwareIAM so
    # older runs that only have DoseAwareIAM state dicts still produce outputs.
    from phase4_evaluation import PRIMARY_MODEL_NAME
    primary = PRIMARY_MODEL_NAME
    state_available = any(
        (MODELS_DIR / f"{primary}_fold{i}.pt").exists() for i in range(10)
    )
    if not state_available:
        legacy = "DoseAwareIAM"
        legacy_available = any((MODELS_DIR / f"{legacy}_fold{i}.pt").exists() for i in range(10))
        if legacy_available:
            print(f"  NOTE: primary={primary!r} has no state dicts under {MODELS_DIR}; falling back to {legacy}")
            primary = legacy
        else:
            print(f"  ERROR: no state dicts found for {primary!r} or {legacy!r}. Run run_primary_canonical.py first.")
            return 1

    sample_map = build_sample_collections(df, lookups, hp, ap, pf, [primary])
    samples = sample_map[primary]
    print(f"  Building OOF attentions from saved {primary} weights ...")
    oof = collect_oof_attention_predictions(primary, ds, samples)
    if not oof.empty:
        oof_metrics = compute_metrics(
            df.iloc[oof["sample_idx"].to_numpy(dtype=int)]["label"].astype(int).to_numpy(),
            oof["prob"].to_numpy(dtype=float),
        )
        print(f"  OOF AUROC={oof_metrics['auroc']:.4f}  AUPRC={oof_metrics['auprc']:.4f}  ({len(oof)} test samples)")
        save_pickle(oof, INTERPRET_DIR / "oof_predictions_with_attention.pkl")
    else:
        print(f"  WARN: OOF dataframe empty (no {primary} weights?), some outputs will be skipped.")

    pair_features, ingredient_index, pathway_t2pw = _resolve_pair_features()
    pw_id2name = _load_pathway_names()
    herb_name, adr_name, formula_name, ing_name = _load_node_metadata()
    f2h = lookups["f2h"]
    dose_map = lookups["dose_map"]

    run_cases_flag = args.mode in {"all", "cases"}
    run_attn_flag = args.mode in {"all", "attn"}
    run_novel_flag = args.mode in {"all", "novel"}
    run_pmda_flag = args.mode in {"all", "pmda"}

    if run_cases_flag:
        run_cases(
            args.n_cases, ds, df, samples, f2h, dose_map, herb_name, adr_name, formula_name, ing_name,
            pair_features, ingredient_index, pathway_t2pw, pw_id2name, oof, INTERPRET_DIR,
        )
    if run_attn_flag:
        run_attention_consistency(df, oof, f2h, herb_name, adr_name, INTERPRET_DIR / "herb_attention_consistency.csv")
    if run_novel_flag:
        run_top_novel_predictions(
            df, oof, f2h, herb_name, adr_name, formula_name, pair_features, dose_map,
            INTERPRET_DIR / "top_novel_predictions.csv", args.top_k_novel,
        )
    if run_pmda_flag:
        custom_path = Path(args.pmda_yaml).resolve() if args.pmda_yaml else None
        interactions = _resolve_pmda_interactions(custom_path)
        run_pmda_concordance(
            interactions, df, oof, f2h, herb_name, adr_name, formula_name,
            INTERPRET_DIR / "pmda_concordance.csv",
        )
    print("Phase 5 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
