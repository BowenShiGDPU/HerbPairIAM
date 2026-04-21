"""Neural models and training loop for HerbPairIAM and its ablations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD

from experiment_utils import (
    OUT_DIR,
    VAL_SELECTION_METRIC,
    compute_metrics,
    compute_peak_memory_mb,
    enable_strict_determinism,
    ensure_output_dirs,
    fold_result_path,
    load_pickle,
    model_state_path,
    save_pickle,
    set_seed,
    val_score_from_metrics,
)

enable_strict_determinism()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILE_DIM = 48
PAIR_FEATURE_KEYS = [
    "convergence",
    "complementarity",
    "union_coverage",
    "ppi_bridge",
    "pw_convergence",
    "pw_complementarity",
    "tissue_coexpr",
    "n_adr_targets",
    "n_h1_adr_targets",
    "n_h2_adr_targets",
    "n_convergence_targets",
]
EDGE_DIM = len(PAIR_FEATURE_KEYS) + 1
NODE_EXTRA_DIM = 3
NODE_DIM = PROFILE_DIM + NODE_EXTRA_DIM
ADR_DIM = PROFILE_DIM
INGREDIENT_EXTRA_DIM = 3
FORMULA_DOSE_DIM = 6


@dataclass
class ModelConfig:
    hidden: int = 32
    dropout: float = 0.3
    lr: float = 5e-3
    wd: float = 1e-4
    epochs: int = 120
    patience: int = 20
    batch_size: int = 32
    seed: int = 42
    neg_ratio: int = 1
    eval_every: int = 2


def load_all():
    return (
        load_pickle(OUT_DIR / "dataset.pkl"),
        load_pickle(OUT_DIR / "herb_target_profiles.pkl"),
        load_pickle(OUT_DIR / "adr_target_profiles.pkl"),
        load_pickle(OUT_DIR / "herb_pair_features.pkl"),
        load_pickle(OUT_DIR / "lookups.pkl"),
    )


def _load_optional_pickle(name: str):
    path = OUT_DIR / name
    return load_pickle(path) if path.exists() else None


def load_optional_artifacts():
    return {
        "ingredient_target_profiles": _load_optional_pickle("ingredient_target_profiles.pkl"),
        "kg_profile_embeddings": _load_optional_pickle("kg_profile_embeddings.pkl"),
    }


def model_profile_backend(model_name: str) -> str:
    if model_name in {"KGEmbedIAM", "IngredientLiteIAM"}:
        return "kg"
    return "svd"


def model_sample_mode(model_name: str) -> str:
    if model_name == "DoseAwareIAM":
        return "doseaware"
    if model_name in {
        "DoseAwareNoDoseGate",
        "DoseAwareHerbOnly",
        "DoseAwarePairOnly",
        "DoseAwareNoADRConditioning",
        "DoseAwareMeanPool",
        "IAM_DoseFeats",
        "DoseAware_ZeroDose",
        "HerbPairIAM",
    }:
        return "doseaware"
    if model_name == "IngredientLiteIAM":
        return "ingredientlite"
    return "baseline"


def model_intrinsic_ablation(model_name: str) -> frozenset[str]:
    """Sample-level ablation tags that are always applied for the given model.

    HerbPairIAM and its historical alias always zero-fill dose-derived
    sample fields; every other model returns an empty frozenset.
    """
    if model_name in {"HerbPairIAM", "DoseAware_ZeroDose"}:
        return frozenset({"AL_dose"})
    return frozenset()


def reduce_profiles(hp, ap, dim=PROFILE_DIM, backend: str = "svd", embedding_bundle: dict | None = None):
    if backend == "kg" and embedding_bundle is not None:
        zeros = np.zeros(dim, dtype=np.float32)
        herb_red = {h: np.asarray(embedding_bundle.get("herb", {}).get(h, zeros), dtype=np.float32) for h in hp}
        adr_red = {a: np.asarray(embedding_bundle.get("adr", {}).get(a, zeros), dtype=np.float32) for a in ap}
        return herb_red, adr_red
    herbs, adrs = sorted(hp.keys()), sorted(ap.keys())
    herb_mat = np.stack([hp[h] for h in herbs])
    adr_mat = np.stack([ap[a] for a in adrs])
    svd = TruncatedSVD(n_components=dim, random_state=42)
    reduced = svd.fit_transform(np.vstack([herb_mat, adr_mat]))
    herb_red = {h: reduced[i].astype(np.float32) for i, h in enumerate(herbs)}
    adr_red = {a: reduced[len(herbs) + i].astype(np.float32) for i, a in enumerate(adrs)}
    return herb_red, adr_red


def _gini(arr) -> float:
    arr = np.sort(np.abs(np.asarray(arr, dtype=np.float32)))
    if arr.size == 0 or float(arr.sum()) == 0.0:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=np.float32)
    return float(((2 * idx - arr.size - 1) * arr).sum() / (arr.size * arr.sum()))


def _pair_dim_for_mode(sample_mode: str) -> int:
    if sample_mode == "doseaware":
        return len(PAIR_FEATURE_KEYS) + 4
    return len(PAIR_FEATURE_KEYS) + 1


def _normalize_ablation_tags(feature_ablation) -> frozenset[str]:
    """Normalise None / single str / iterable-of-str into a frozenset of tags."""

    if feature_ablation is None:
        return frozenset()
    if isinstance(feature_ablation, str):
        return frozenset({feature_ablation})
    return frozenset(feature_ablation)


def _ablate_pair_feature_dict(pf: dict, feature_ablation) -> dict:
    """Zero out per-pair feature channels driven by the active ablation tags."""

    tags = _normalize_ablation_tags(feature_ablation)
    out = {k: float(pf.get(k, 0.0)) for k in PAIR_FEATURE_KEYS}

    if "AL_pair_all" in tags:
        for k in PAIR_FEATURE_KEYS:
            out[k] = 0.0
        return out

    pathway_tags = {"without_pathway", "AL_pair_multiomics"}
    tissue_tags = {"without_tissue", "AL_pair_multiomics"}
    ppi_tags = {"without_ppi", "AL_pair_direct"}
    convergence_tags = {"without_convergence", "AL_pair_direct"}
    complementarity_tags = {"without_complementarity", "AL_pair_direct"}
    union_tags = {"without_union_coverage", "AL_pair_direct"}

    if tags & pathway_tags:
        out["pw_convergence"] = 0.0
        out["pw_complementarity"] = 0.0
    if tags & tissue_tags:
        out["tissue_coexpr"] = 0.0
    if tags & ppi_tags:
        out["ppi_bridge"] = 0.0
    if tags & convergence_tags:
        out["convergence"] = 0.0
        out["n_convergence_targets"] = 0.0
    if tags & complementarity_tags:
        out["complementarity"] = 0.0
    if tags & union_tags:
        out["union_coverage"] = 0.0
    return out


def _build_formula_dose_features(dose_ratio_arr: np.ndarray, dose_abs_arr: np.ndarray) -> np.ndarray:
    total_dose = float(dose_abs_arr.sum()) if dose_abs_arr.size else 0.0
    mean_dose = float(dose_abs_arr.mean()) if dose_abs_arr.size else 0.0
    max_dose = float(dose_abs_arr.max()) if dose_abs_arr.size else 0.0
    dose_std = float(dose_ratio_arr.std()) if dose_ratio_arr.size > 1 else 0.0
    dose_gini = _gini(dose_ratio_arr) if dose_ratio_arr.size > 1 else 0.0
    max_ratio = float(dose_ratio_arr.max()) if dose_ratio_arr.size else 0.0
    return np.asarray(
        [np.log1p(total_dose), np.log1p(mean_dose), np.log1p(max_dose), dose_std, dose_gini, max_ratio],
        dtype=np.float32,
    )


def build_sample_collections(
    df,
    lookups,
    herb_profiles,
    adr_profiles,
    pair_features,
    model_names,
    optional_artifacts: dict | None = None,
    feature_ablation=None,
):
    optional_artifacts = optional_artifacts or load_optional_artifacts()
    ingredient_profiles = optional_artifacts.get("ingredient_target_profiles")
    kg_bundle = optional_artifacts.get("kg_profile_embeddings")
    cache = {}
    sample_map = {}
    caller_ablation = _normalize_ablation_tags(feature_ablation)
    for model_name in sorted(set(model_names)):
        profile_backend = model_profile_backend(model_name)
        sample_mode = model_sample_mode(model_name)
        combined_ablation = caller_ablation | model_intrinsic_ablation(model_name)
        cache_key = (profile_backend, sample_mode, combined_ablation)
        if cache_key not in cache:
            herb_red, adr_red = reduce_profiles(
                herb_profiles,
                adr_profiles,
                backend=profile_backend,
                embedding_bundle=kg_bundle,
            )
            ingredient_red = None
            if kg_bundle is not None:
                ingredient_red = kg_bundle.get("ingredient")
            cache[cache_key] = precompute_samples(
                df,
                lookups,
                herb_profiles,
                adr_profiles,
                herb_red,
                adr_red,
                pair_features,
                sample_mode=sample_mode,
                ingredient_profiles=ingredient_profiles,
                ingredient_red=ingredient_red,
                feature_ablation=combined_ablation if combined_ablation else None,
            )
        sample_map[model_name] = cache[cache_key]
    return sample_map


def precompute_samples(
    df,
    lookups,
    herb_profiles,
    adr_profiles,
    herb_red,
    adr_red,
    pair_features,
    sample_mode: str = "baseline",
    ingredient_profiles: dict | None = None,
    ingredient_red: dict | None = None,
    feature_ablation=None,
):
    f2h = lookups["f2h"]
    dose_map = lookups["dose_map"]
    h2i = lookups.get("h2i", {})
    samples = []

    ablation_tags = _normalize_ablation_tags(feature_ablation)
    use_without_dose = bool(ablation_tags & {"without_dose", "AL_dose"})
    use_without_individual = "AL_individual" in ablation_tags

    for row in df.itertuples(index=False):
        f_id = row.TCMF_id
        a_id = row.Adr_id
        herbs = sorted(f2h.get(f_id, []))
        adr_vec = adr_red.get(a_id)
        adr_profile = adr_profiles.get(a_id)

        if len(herbs) == 0 or adr_vec is None or adr_profile is None:
            samples.append(None)
            continue

        uniform_dr = 1.0 / max(len(herbs), 1)
        dose_ratios = np.asarray(
            [
                (uniform_dr if use_without_dose else float(dose_map.get((f_id, h), {}).get("dose_ratio", uniform_dr)))
                for h in herbs
            ],
            dtype=np.float32,
        )
        dose_abs = np.asarray(
            [0.0 if use_without_dose else float(dose_map.get((f_id, h), {}).get("dose_g", 0.0)) for h in herbs],
            dtype=np.float32,
        )
        formula_dose_features = (
            np.zeros(FORMULA_DOSE_DIM, dtype=np.float32)
            if use_without_dose
            else _build_formula_dose_features(dose_ratios, dose_abs)
        )
        node_features = []
        pair_src = []
        pair_dst = []
        pair_feat_list = []
        edge_src = []
        edge_dst = []
        edge_feat_list = []
        ingredient_feat_list = []

        for i, h in enumerate(herbs):
            dr = uniform_dr if use_without_dose else dose_map.get((f_id, h), {}).get("dose_ratio", uniform_dr)
            dose_g = 0.0 if use_without_dose else dose_map.get((f_id, h), {}).get("dose_g", 0.0)
            raw_profile = herb_profiles.get(h, np.zeros_like(adr_profile))
            red_profile = herb_red.get(h, np.zeros(PROFILE_DIM, dtype=np.float32))
            overlap = 0.0 if use_without_individual else float(np.dot(raw_profile, adr_profile))
            node_tail = [dr, overlap, overlap * dr]
            if sample_mode == "doseaware":
                node_tail.extend([dose_g, np.log1p(dose_g), dr * np.log1p(dose_g)])
            node_profile_term = (
                np.zeros(PROFILE_DIM, dtype=np.float32)
                if use_without_individual
                else red_profile * dr
            )
            node_features.append(
                np.concatenate(
                    [
                        node_profile_term,
                        np.asarray(node_tail, dtype=np.float32),
                    ]
                )
            )
            if sample_mode == "ingredientlite" and ingredient_profiles is not None:
                ingredients = sorted(h2i.get(h, set()))
                if ingredients:
                    ing_weight = float(dr) / max(len(ingredients), 1)
                    for ing in ingredients:
                        ing_profile = ingredient_profiles.get(ing)
                        if ing_profile is None:
                            continue
                        ing_red = (ingredient_red or {}).get(ing, np.zeros(PROFILE_DIM, dtype=np.float32))
                        ing_overlap = float(np.dot(ing_profile, adr_profile))
                        ingredient_feat_list.append(
                            np.concatenate(
                                [
                                    np.asarray(ing_red, dtype=np.float32) * ing_weight,
                                    np.asarray([ing_weight, ing_overlap, ing_overlap * ing_weight], dtype=np.float32),
                                ]
                            )
                        )

        for i, h1 in enumerate(herbs):
            for j in range(i + 1, len(herbs)):
                h2 = herbs[j]
                pf = _ablate_pair_feature_dict(pair_features.get((h1, h2, a_id), {}), feature_ablation)
                ef = [float(pf.get(k, 0.0)) for k in PAIR_FEATURE_KEYS]
                dr1 = uniform_dr if use_without_dose else dose_map.get((f_id, h1), {}).get("dose_ratio", uniform_dr)
                dr2 = uniform_dr if use_without_dose else dose_map.get((f_id, h2), {}).get("dose_ratio", uniform_dr)
                dose_product = 0.0 if use_without_dose else float(dr1 * dr2)
                ef.append(dose_product)
                if sample_mode == "doseaware":
                    if use_without_dose:
                        ef.extend([0.0, 0.0, 0.0])
                    else:
                        ef.extend([float(abs(dr1 - dr2)), float(max(dr1, dr2)), float(min(dr1, dr2))])
                pair_src.append(i)
                pair_dst.append(j)
                pair_feat_list.append(ef)
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_feat_list.extend([ef, ef])

        sample = {
            "node_features": torch.tensor(np.asarray(node_features), dtype=torch.float32, device=DEVICE),
            "adr_features": torch.tensor(np.asarray(adr_vec), dtype=torch.float32, device=DEVICE),
            "formula_dose_features": torch.tensor(formula_dose_features, dtype=torch.float32, device=DEVICE),
            "pair_src": torch.tensor(pair_src, dtype=torch.long, device=DEVICE),
            "pair_dst": torch.tensor(pair_dst, dtype=torch.long, device=DEVICE),
            "pair_features": torch.tensor(pair_feat_list, dtype=torch.float32, device=DEVICE)
            if pair_feat_list
            else torch.zeros((0, _pair_dim_for_mode(sample_mode)), dtype=torch.float32, device=DEVICE),
            "edge_src": torch.tensor(edge_src, dtype=torch.long, device=DEVICE),
            "edge_dst": torch.tensor(edge_dst, dtype=torch.long, device=DEVICE),
            "edge_features": torch.tensor(edge_feat_list, dtype=torch.float32, device=DEVICE)
            if edge_feat_list
            else torch.zeros((0, _pair_dim_for_mode(sample_mode)), dtype=torch.float32, device=DEVICE),
            "ingredient_features": torch.tensor(np.asarray(ingredient_feat_list), dtype=torch.float32, device=DEVICE)
            if ingredient_feat_list
            else torch.zeros((0, PROFILE_DIM + INGREDIENT_EXTRA_DIM), dtype=torch.float32, device=DEVICE),
            "n_herbs": len(herbs),
            "herbs": herbs,
            "tcmf_id": f_id,
            "adr_id": a_id,
            "sample_mode": sample_mode,
        }
        samples.append(sample)

    return samples


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HerbInteractionGraph(nn.Module):
    def __init__(
        self,
        node_in=NODE_DIM,
        edge_in=EDGE_DIM,
        adr_in=ADR_DIM,
        hidden=32,
        dropout=0.3,
        mp_layers=1,
        adr_conditioned=True,
        readout="attention",
        scorer="mlp",
    ):
        super().__init__()
        self.mp_layers = mp_layers
        self.adr_conditioned = adr_conditioned
        self.readout = readout
        self.scorer = scorer

        self.node_enc = MLP(node_in, hidden, hidden, dropout)
        self.edge_enc = MLP(edge_in, hidden, hidden, dropout)
        self.adr_enc = MLP(adr_in, hidden, hidden, dropout)
        self.msg_layers = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(mp_layers)])
        self.update_layers = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(mp_layers)])
        attn_in = hidden * 2 if adr_conditioned else hidden
        self.attn = nn.Linear(attn_in, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dot_proj = nn.Linear(hidden, hidden) if scorer == "dot" else None

    def _readout(self, h, a):
        if self.readout == "sum":
            z = h.sum(dim=0)
            alpha = torch.full((h.size(0),), 1.0 / max(h.size(0), 1), device=h.device)
            return z, alpha
        if self.readout == "mean":
            z = h.mean(dim=0)
            alpha = torch.full((h.size(0),), 1.0 / max(h.size(0), 1), device=h.device)
            return z, alpha

        if self.adr_conditioned:
            a_exp = a.unsqueeze(0).expand(h.size(0), -1)
            logits = self.attn(torch.cat([h, a_exp], dim=-1)).squeeze(-1)
        else:
            logits = self.attn(h).squeeze(-1)
        alpha = F.softmax(logits, dim=0)
        z = (alpha.unsqueeze(-1) * h).sum(dim=0)
        return z, alpha

    def forward(self, sample):
        h = self.node_enc(sample["node_features"])
        a = self.adr_enc(sample["adr_features"])
        ef = self.edge_enc(sample["edge_features"]) if sample["edge_features"].numel() > 0 else sample["edge_features"]

        for msg_layer, update_layer in zip(self.msg_layers, self.update_layers):
            if sample["edge_src"].numel() == 0:
                break
            msg_input = torch.cat([h[sample["edge_src"]], ef], dim=-1)
            msg = msg_layer(msg_input)
            agg = torch.zeros_like(h)
            agg.index_add_(0, sample["edge_dst"], msg)
            h = F.relu(update_layer(torch.cat([h, agg], dim=-1)))

        z, alpha = self._readout(h, a)
        if self.scorer == "dot":
            logit = torch.dot(self.dot_proj(z), a)
        else:
            logit = self.pred(torch.cat([z, a], dim=-1)).squeeze(-1)
        return logit, {"herb_attn": alpha.detach()}


class InteractionAwareSetModel(nn.Module):
    def __init__(self, node_in=NODE_DIM, pair_in=EDGE_DIM, adr_in=ADR_DIM, hidden=32, dropout=0.3):
        super().__init__()
        self.node_enc = MLP(node_in, hidden, hidden, dropout)
        self.pair_enc = MLP(pair_in, hidden, hidden, dropout)
        self.adr_enc = MLP(adr_in, hidden, hidden, dropout)
        self.node_attn = nn.Linear(hidden * 2, 1)
        self.pair_attn = nn.Linear(hidden * 2, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sample):
        node_h = self.node_enc(sample["node_features"])
        a = self.adr_enc(sample["adr_features"])
        a_exp = a.unsqueeze(0).expand(node_h.size(0), -1)
        node_alpha = F.softmax(self.node_attn(torch.cat([node_h, a_exp], dim=-1)).squeeze(-1), dim=0)
        z_node = (node_alpha.unsqueeze(-1) * node_h).sum(dim=0)

        if sample["pair_features"].shape[0] > 0:
            pair_h = self.pair_enc(sample["pair_features"])
            a_pair = a.unsqueeze(0).expand(pair_h.size(0), -1)
            pair_alpha = F.softmax(self.pair_attn(torch.cat([pair_h, a_pair], dim=-1)).squeeze(-1), dim=0)
            z_pair = (pair_alpha.unsqueeze(-1) * pair_h).sum(dim=0)
        else:
            pair_alpha = torch.zeros((0,), device=node_h.device)
            z_pair = torch.zeros_like(z_node)

        logit = self.pred(torch.cat([z_node, z_pair, a], dim=-1)).squeeze(-1)
        return logit, {"herb_attn": node_alpha.detach(), "pair_attn": pair_alpha.detach()}


class DoseAwareInteractionAwareSetModel(nn.Module):
    def __init__(
        self,
        node_in,
        pair_in,
        adr_in,
        dose_in,
        hidden=32,
        dropout=0.3,
        adr_conditioned=True,
        use_dose_gate=True,
        use_herb_branch=True,
        use_pair_branch=True,
        pool_type="attention",
    ):
        super().__init__()
        self.adr_conditioned = adr_conditioned
        self.use_dose_gate = use_dose_gate
        self.use_herb_branch = use_herb_branch
        self.use_pair_branch = use_pair_branch
        self.pool_type = pool_type
        self.node_enc = MLP(node_in, hidden, hidden, dropout)
        self.pair_enc = MLP(pair_in, hidden, hidden, dropout)
        self.adr_enc = MLP(adr_in, hidden, hidden, dropout)
        self.dose_enc = MLP(dose_in, hidden, hidden, dropout)
        self.node_attn = nn.Linear(hidden * 2, 1)
        self.pair_attn = nn.Linear(hidden * 2, 1)
        self.node_attn_noadr = nn.Linear(hidden, 1)
        self.pair_attn_noadr = nn.Linear(hidden, 1)
        self.node_gate = nn.Linear(hidden * 2, hidden)
        self.pair_gate = nn.Linear(hidden * 2, hidden)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sample):
        node_h = self.node_enc(sample["node_features"])
        a = self.adr_enc(sample["adr_features"])
        dose_z = self.dose_enc(sample["formula_dose_features"])
        if self.use_herb_branch:
            if self.pool_type == "mean":
                node_alpha = torch.full((node_h.size(0),), 1.0 / max(node_h.size(0), 1), device=node_h.device)
            else:
                if self.adr_conditioned:
                    a_exp = a.unsqueeze(0).expand(node_h.size(0), -1)
                    node_alpha = F.softmax(self.node_attn(torch.cat([node_h, a_exp], dim=-1)).squeeze(-1), dim=0)
                else:
                    node_alpha = F.softmax(self.node_attn_noadr(node_h).squeeze(-1), dim=0)
            z_node = (node_alpha.unsqueeze(-1) * node_h).sum(dim=0)
            if self.use_dose_gate:
                node_gate = torch.sigmoid(self.node_gate(torch.cat([a, dose_z], dim=-1)))
                z_node = z_node * node_gate
        else:
            node_alpha = torch.zeros((0,), device=node_h.device)
            z_node = torch.zeros(node_h.size(-1), device=node_h.device)

        if self.use_pair_branch and sample["pair_features"].shape[0] > 0:
            pair_h = self.pair_enc(sample["pair_features"])
            if self.pool_type == "mean":
                pair_alpha = torch.full((pair_h.size(0),), 1.0 / max(pair_h.size(0), 1), device=pair_h.device)
            else:
                if self.adr_conditioned:
                    a_pair = a.unsqueeze(0).expand(pair_h.size(0), -1)
                    pair_alpha = F.softmax(self.pair_attn(torch.cat([pair_h, a_pair], dim=-1)).squeeze(-1), dim=0)
                else:
                    pair_alpha = F.softmax(self.pair_attn_noadr(pair_h).squeeze(-1), dim=0)
            z_pair = (pair_alpha.unsqueeze(-1) * pair_h).sum(dim=0)
            if self.use_dose_gate:
                pair_gate = torch.sigmoid(self.pair_gate(torch.cat([a, dose_z], dim=-1)))
                z_pair = z_pair * pair_gate
        else:
            pair_alpha = torch.zeros((0,), device=node_h.device)
            z_pair = torch.zeros_like(z_node)

        logit = self.pred(torch.cat([z_node, z_pair, dose_z, a], dim=-1)).squeeze(-1)
        return logit, {"herb_attn": node_alpha.detach(), "pair_attn": pair_alpha.detach(), "dose_gate": dose_z.detach()}


class IngredientLiteInteractionAwareSetModel(nn.Module):
    def __init__(self, node_in, pair_in, adr_in, ingredient_in, hidden=32, dropout=0.3):
        super().__init__()
        self.node_enc = MLP(node_in, hidden, hidden, dropout)
        self.pair_enc = MLP(pair_in, hidden, hidden, dropout)
        self.ingredient_enc = MLP(ingredient_in, hidden, hidden, dropout)
        self.adr_enc = MLP(adr_in, hidden, hidden, dropout)
        self.node_attn = nn.Linear(hidden * 2, 1)
        self.pair_attn = nn.Linear(hidden * 2, 1)
        self.ingredient_attn = nn.Linear(hidden * 2, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sample):
        node_h = self.node_enc(sample["node_features"])
        a = self.adr_enc(sample["adr_features"])
        a_exp = a.unsqueeze(0).expand(node_h.size(0), -1)
        node_alpha = F.softmax(self.node_attn(torch.cat([node_h, a_exp], dim=-1)).squeeze(-1), dim=0)
        z_node = (node_alpha.unsqueeze(-1) * node_h).sum(dim=0)

        if sample["pair_features"].shape[0] > 0:
            pair_h = self.pair_enc(sample["pair_features"])
            a_pair = a.unsqueeze(0).expand(pair_h.size(0), -1)
            pair_alpha = F.softmax(self.pair_attn(torch.cat([pair_h, a_pair], dim=-1)).squeeze(-1), dim=0)
            z_pair = (pair_alpha.unsqueeze(-1) * pair_h).sum(dim=0)
        else:
            pair_alpha = torch.zeros((0,), device=node_h.device)
            z_pair = torch.zeros_like(z_node)

        if sample["ingredient_features"].shape[0] > 0:
            ing_h = self.ingredient_enc(sample["ingredient_features"])
            a_ing = a.unsqueeze(0).expand(ing_h.size(0), -1)
            ing_alpha = F.softmax(self.ingredient_attn(torch.cat([ing_h, a_ing], dim=-1)).squeeze(-1), dim=0)
            z_ing = (ing_alpha.unsqueeze(-1) * ing_h).sum(dim=0)
        else:
            ing_alpha = torch.zeros((0,), device=node_h.device)
            z_ing = torch.zeros_like(z_node)

        logit = self.pred(torch.cat([z_node, z_pair, z_ing, a], dim=-1)).squeeze(-1)
        return logit, {"herb_attn": node_alpha.detach(), "pair_attn": pair_alpha.detach(), "ingredient_attn": ing_alpha.detach()}


class AttentionPoolNoMP(HerbInteractionGraph):
    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["mp_layers"] = 0
        super().__init__(**kwargs)


def _first_sample_dims(sample_example):
    node_in = int(sample_example["node_features"].shape[-1]) if sample_example is not None else NODE_DIM
    pair_in = int(sample_example["pair_features"].shape[-1]) if sample_example is not None else EDGE_DIM
    adr_in = int(sample_example["adr_features"].shape[-1]) if sample_example is not None else ADR_DIM
    ingredient_in = int(sample_example.get("ingredient_features", torch.zeros((0, PROFILE_DIM + INGREDIENT_EXTRA_DIM))).shape[-1])
    dose_in = int(sample_example.get("formula_dose_features", torch.zeros(FORMULA_DOSE_DIM)).shape[-1])
    return node_in, pair_in, adr_in, ingredient_in, dose_in


def build_model(model_name: str, config: ModelConfig, sample_example=None, vocab: dict | None = None):
    """Construct a neural model by name.

    Parameters
    ----------
    vocab:
        Optional mapping with keys ``"herb_vocab"`` and ``"adr_vocab"``, each
        a list of canonical string ids. Required only by learnable-embedding
        ablations (e.g. ``HerbEmbIAM``); ignored by all KG-feature models.
        Passing a ``lookups`` dict (from :func:`load_all`) also works — we
        will fall back to ``lookups["h2i"]`` / ``lookups["adr_nodes"]`` when
        the explicit vocab keys are missing.
    """
    node_in, pair_in, adr_in, ingredient_in, dose_in = _first_sample_dims(sample_example)
    if model_name == "HerbEmbIAM":
        # Import is deferred so the main training pipeline doesn't pay the
        # import cost unless the ablation is actually requested.
        from models.herb_emb_iam import HerbEmbIAM  # type: ignore[import-untyped]
        if vocab is None:
            raise ValueError(
                "HerbEmbIAM requires vocab={'herb_vocab': [...], 'adr_vocab': [...]}"
            )
        herb_vocab = vocab.get("herb_vocab")
        adr_vocab = vocab.get("adr_vocab")
        if herb_vocab is None or adr_vocab is None:
            raise ValueError("vocab missing 'herb_vocab' or 'adr_vocab'")
        return HerbEmbIAM(
            herb_vocab=list(herb_vocab),
            adr_vocab=list(adr_vocab),
            pair_in=pair_in,
            hidden=config.hidden,
            emb_dim=PROFILE_DIM,
            dropout=config.dropout,
        ).to(DEVICE)
    if model_name == "HerbInteractionGraph":
        return HerbInteractionGraph(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout).to(DEVICE)
    if model_name == "AttentionPool_NoMP":
        return AttentionPoolNoMP(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout).to(DEVICE)
    if model_name in {"InteractionAwareSetModel", "KGEmbedIAM", "IAM_DoseFeats"}:
        return InteractionAwareSetModel(node_in=node_in, pair_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout).to(DEVICE)
    if model_name == "IAM_Wide":
        return InteractionAwareSetModel(node_in=node_in, pair_in=pair_in, adr_in=adr_in, hidden=44, dropout=config.dropout).to(DEVICE)
    if model_name in {"DoseAwareIAM", "DoseAware_ZeroDose", "HerbPairIAM"}:
        return DoseAwareInteractionAwareSetModel(node_in=node_in, pair_in=pair_in, adr_in=adr_in, dose_in=dose_in, hidden=config.hidden, dropout=config.dropout).to(DEVICE)
    if model_name == "DoseAwareNoDoseGate":
        return DoseAwareInteractionAwareSetModel(
            node_in=node_in,
            pair_in=pair_in,
            adr_in=adr_in,
            dose_in=dose_in,
            hidden=config.hidden,
            dropout=config.dropout,
            use_dose_gate=False,
        ).to(DEVICE)
    if model_name == "DoseAwareHerbOnly":
        return DoseAwareInteractionAwareSetModel(
            node_in=node_in,
            pair_in=pair_in,
            adr_in=adr_in,
            dose_in=dose_in,
            hidden=config.hidden,
            dropout=config.dropout,
            use_pair_branch=False,
        ).to(DEVICE)
    if model_name == "DoseAwarePairOnly":
        return DoseAwareInteractionAwareSetModel(
            node_in=node_in,
            pair_in=pair_in,
            adr_in=adr_in,
            dose_in=dose_in,
            hidden=config.hidden,
            dropout=config.dropout,
            use_herb_branch=False,
        ).to(DEVICE)
    if model_name == "DoseAwareNoADRConditioning":
        return DoseAwareInteractionAwareSetModel(
            node_in=node_in,
            pair_in=pair_in,
            adr_in=adr_in,
            dose_in=dose_in,
            hidden=config.hidden,
            dropout=config.dropout,
            adr_conditioned=False,
        ).to(DEVICE)
    if model_name == "DoseAwareMeanPool":
        return DoseAwareInteractionAwareSetModel(
            node_in=node_in,
            pair_in=pair_in,
            adr_in=adr_in,
            dose_in=dose_in,
            hidden=config.hidden,
            dropout=config.dropout,
            pool_type="mean",
        ).to(DEVICE)
    if model_name == "IngredientLiteIAM":
        return IngredientLiteInteractionAwareSetModel(
            node_in=node_in,
            pair_in=pair_in,
            adr_in=adr_in,
            ingredient_in=ingredient_in,
            hidden=config.hidden,
            dropout=config.dropout,
        ).to(DEVICE)
    if model_name == "SumPool":
        return HerbInteractionGraph(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout, mp_layers=0, readout="sum").to(DEVICE)
    if model_name == "MeanPool":
        return HerbInteractionGraph(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout, mp_layers=0, readout="mean").to(DEVICE)
    if model_name == "NoADRConditioning":
        return HerbInteractionGraph(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout, adr_conditioned=False).to(DEVICE)
    if model_name == "DotScorer":
        return HerbInteractionGraph(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout, scorer="dot").to(DEVICE)
    if model_name == "TwoLayerMP":
        return HerbInteractionGraph(node_in=node_in, edge_in=pair_in, adr_in=adr_in, hidden=config.hidden, dropout=config.dropout, mp_layers=2).to(DEVICE)
    raise ValueError(f"Unknown model_name: {model_name}")


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


def predict_indices(model, samples, labels, indices):
    model.eval()
    preds = []
    labs = []
    attentions = []
    indices_out = []
    t0 = time.time()
    with torch.no_grad():
        for idx in indices:
            sample = samples[int(idx)]
            if sample is None:
                continue
            logit, aux = model(sample)
            preds.append(torch.sigmoid(logit).item())
            labs.append(int(labels[int(idx)]))
            attentions.append(aux)
            indices_out.append(int(idx))
    elapsed = time.time() - t0
    return {
        "indices": np.asarray(indices_out, dtype=int),
        "y_true": np.asarray(labs, dtype=int),
        "y_prob": np.asarray(preds, dtype=float),
        "attentions": attentions,
        "elapsed_sec": elapsed,
    }


def train_one_split(
    model_name: str,
    samples,
    labels,
    split: dict,
    config: ModelConfig,
    save_model: bool = True,
    vocab: dict | None = None,
):
    """Train one outer-CV fold.

    Extra ``vocab`` kwarg is forwarded to ``build_model`` so that
    learnable-embedding ablations (e.g. ``HerbEmbIAM``) can receive the
    herb / ADR vocabulary without changing the shared training plumbing.
    """
    ensure_output_dirs()
    set_seed(config.seed + int(split["fold"]) if "fold" in split else config.seed)
    labels = np.asarray(labels, dtype=int)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)

    sample_example = next((sample for sample in samples if sample is not None), None)
    model = build_model(model_name, config, sample_example=sample_example, vocab=vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    best_state = None
    best_val_score = -np.inf
    patience_steps = 0
    train_start = time.time()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        batch_counter = 0
        balanced_idx = balance_train_indices(labels, train_idx, neg_ratio=config.neg_ratio, seed=config.seed + epoch)
        for idx in balanced_idx:
            sample = samples[int(idx)]
            if sample is None:
                continue
            logit, _ = model(sample)
            target = torch.tensor(float(labels[int(idx)]), dtype=torch.float32, device=DEVICE)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            (loss / config.batch_size).backward()
            batch_counter += 1
            if batch_counter % config.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
        if batch_counter % config.batch_size != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % config.eval_every == 0:
            val_pred = predict_indices(model, samples, labels, val_idx)
            val_metrics = compute_metrics(val_pred["y_true"], val_pred["y_prob"])
            val_score = val_score_from_metrics(val_metrics)
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_steps = 0
            else:
                patience_steps += 1
            if patience_steps >= max(1, config.patience // config.eval_every):
                break

    train_time = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = predict_indices(model, samples, labels, val_idx)
    val_metrics = compute_metrics(val_pred["y_true"], val_pred["y_prob"])
    test_pred = predict_indices(model, samples, labels, test_idx)
    test_metrics = compute_metrics(test_pred["y_true"], test_pred["y_prob"], threshold=val_metrics["threshold"])

    state_path = None
    if save_model:
        fold_id = split.get("fold", split.get("seed", 0))
        state_path = model_state_path(model_name, int(fold_id))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), state_path)

    inference_time_sec = test_pred["elapsed_sec"]
    fold_result = {
        "model": model_name,
        "fold": int(split.get("fold", split.get("seed", 0))),
        "split_key": "fold" if "fold" in split else "seed",
        "y_true": test_pred["y_true"],
        "y_prob": test_pred["y_prob"],
        "test_indices": test_pred["indices"],
        "y_true_val": val_pred["y_true"],
        "y_prob_val": val_pred["y_prob"],
        "val_indices": val_pred["indices"],
        "val_auroc": float(val_metrics["auroc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "threshold": float(val_metrics["threshold"]),
        **{k: float(v) for k, v in test_metrics.items()},
        "train_time_sec": float(train_time),
        "inference_time_sec": float(inference_time_sec),
        "peak_memory_mb": float(compute_peak_memory_mb()),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "model_state_path": str(state_path) if state_path is not None else None,
        "config": config.__dict__.copy(),
    }
    return fold_result


def run_cv(model_name: str, samples, labels, fold_splits, config: ModelConfig, fold_subset: list[int] | None = None):
    results = []
    for split in fold_splits:
        split_id = int(split.get("fold", split.get("seed", 0)))
        if fold_subset is not None and split_id not in fold_subset:
            continue
        result = train_one_split(model_name, samples, labels, split, config)
        save_pickle(result, fold_result_path(model_name, split_id))
        results.append(result)
    return results


def summarize_results(results: list[dict]):
    metrics = {}
    for key in ["auroc", "auprc", "precision", "recall", "f1", "mcc", "train_time_sec", "inference_time_sec"]:
        vals = np.asarray([float(r[key]) for r in results], dtype=float)
        metrics[f"{key}_mean"] = float(vals.mean())
        metrics[f"{key}_std"] = float(vals.std(ddof=0))
    return metrics


def search_config(model_name: str, samples, labels, split: dict, candidate_configs: list[ModelConfig]):
    best_config = candidate_configs[0]
    best_score = -np.inf
    val_key = f"val_{VAL_SELECTION_METRIC}"
    for config in candidate_configs:
        result = train_one_split(model_name, samples, labels, split, config, save_model=False)
        score = float(result[val_key])
        if score > best_score:
            best_score = score
            best_config = config
    return best_config
