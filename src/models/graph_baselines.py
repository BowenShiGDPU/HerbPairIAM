"""
End-to-end graph baselines for Formula-ADR prediction.

Implements:
- R-GCN on the bio-KG without Formula nodes
- HGT on the bio-KG without Formula nodes
- R-GCN with Formula nodes and train-only Formula-ADR encoder edges
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, RGCNConv

from experiment_utils import (
    OUT_DIR,
    compute_metrics,
    compute_peak_memory_mb,
    enable_strict_determinism,
    fold_result_path,
    model_state_path,
    save_pickle,
    set_seed,
    val_score_from_metrics,
)

# See comment in neural_models.py — opt into strict determinism once per
# process. Graph baselines (R-GCN, HGT) also benefit from deterministic
# cuDNN for message passing.
enable_strict_determinism()


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "final_data_clean"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GraphConfig:
    hidden: int = 64
    layers: int = 2
    heads: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    wd: float = 1e-4
    epochs: int = 120
    patience: int = 20
    batch_size: int = 256
    seed: int = 42
    neg_ratio: int = 1
    eval_every: int = 2


def load_tables():
    return {
        "formula_herb": pd.read_csv(DATA_DIR / "CMM_TCMF.csv"),
        "herb_ingredient": pd.read_csv(DATA_DIR / "CMM_Ingredient.csv"),
        "ingredient_target": pd.read_csv(DATA_DIR / "Ingredient_Target_.csv"),
        "herb_target": pd.read_csv(DATA_DIR / "CMM_Target.csv"),
        "target_target": pd.read_csv(DATA_DIR / "Target_Target.csv"),
        "adr_target": pd.read_csv(DATA_DIR / "ADR_Target.csv"),
        "cmm_nodes": pd.read_csv(DATA_DIR / "CMM_nodes.csv"),
        "ingredient_nodes": pd.read_csv(DATA_DIR / "Ingredient_nodes.csv"),
        "target_nodes": pd.read_csv(DATA_DIR / "Target_nodes.csv"),
        "adr_nodes": pd.read_csv(DATA_DIR / "ADR_nodes.csv"),
        "tcmf_nodes": pd.read_csv(DATA_DIR / "TCMF_nodes.csv"),
    }


def make_type_maps(tables, include_formula=False):
    type_maps = {
        "herb": {x: i for i, x in enumerate(sorted(tables["cmm_nodes"]["CMM_id"].unique().tolist()))},
        "ingredient": {x: i for i, x in enumerate(sorted(tables["ingredient_nodes"]["In_id"].unique().tolist()))},
        "target": {x: i for i, x in enumerate(sorted(tables["target_nodes"]["Ta_id"].unique().tolist()))},
        "adr": {x: i for i, x in enumerate(sorted(tables["adr_nodes"]["Adr_id"].unique().tolist()))},
    }
    if include_formula:
        type_maps["formula"] = {x: i for i, x in enumerate(sorted(tables["tcmf_nodes"]["TCMF_id"].unique().tolist()))}
    return type_maps


def build_formula_lookup(tables, herb_map):
    formula_to_herbs = {}
    formula_to_weights = {}
    for f_id, group in tables["formula_herb"].groupby("TCMF_id"):
        herbs = [herb_map[h] for h in group["CMM_id"].tolist() if h in herb_map]
        weights = group["dose_ratio"].fillna(1.0 / max(len(group), 1)).astype(float).tolist()
        formula_to_herbs[f_id] = herbs
        formula_to_weights[f_id] = weights
    return formula_to_herbs, formula_to_weights


def _add_edges(edge_dict, relation, src_list, dst_list):
    if len(src_list) == 0:
        return
    edge_dict.setdefault(relation, [[], []])
    edge_dict[relation][0].extend(src_list)
    edge_dict[relation][1].extend(dst_list)


def build_rgcn_graph(tables, type_maps, split=None, include_formula=False):
    order = ["herb", "ingredient", "target", "adr"] + (["formula"] if include_formula else [])
    offsets = {}
    total_nodes = 0
    for node_type in order:
        offsets[node_type] = total_nodes
        total_nodes += len(type_maps[node_type])

    edge_dict = {}
    herb_map = type_maps["herb"]
    ing_map = type_maps["ingredient"]
    target_map = type_maps["target"]
    adr_map = type_maps["adr"]

    hf = tables["herb_ingredient"]
    src = [offsets["herb"] + herb_map[h] for h in hf["CMM_id"] if h in herb_map]
    dst = [offsets["ingredient"] + ing_map[i] for i in hf["In_id"] if i in ing_map]
    valid = [(offsets["herb"] + herb_map[h], offsets["ingredient"] + ing_map[i]) for h, i in zip(hf["CMM_id"], hf["In_id"]) if h in herb_map and i in ing_map]
    _add_edges(edge_dict, "herb_to_ingredient", [a for a, _ in valid], [b for _, b in valid])
    _add_edges(edge_dict, "ingredient_to_herb", [b for a, b in valid], [a for a, b in valid])

    it = tables["ingredient_target"]
    valid = [(offsets["ingredient"] + ing_map[i], offsets["target"] + target_map[t]) for i, t in zip(it["In_id"], it["Ta_id"]) if i in ing_map and t in target_map]
    _add_edges(edge_dict, "ingredient_to_target", [a for a, _ in valid], [b for _, b in valid])
    _add_edges(edge_dict, "target_to_ingredient", [b for a, b in valid], [a for a, b in valid])

    ht = tables["herb_target"]
    valid = [(offsets["herb"] + herb_map[h], offsets["target"] + target_map[t]) for h, t in zip(ht["CMM_id"], ht["Ta_id"]) if h in herb_map and t in target_map]
    _add_edges(edge_dict, "herb_to_target", [a for a, _ in valid], [b for _, b in valid])
    _add_edges(edge_dict, "target_to_herb", [b for a, b in valid], [a for a, b in valid])

    tt = tables["target_target"]
    valid = [(offsets["target"] + target_map[t1], offsets["target"] + target_map[t2]) for t1, t2 in zip(tt["Ta_id_1"], tt["Ta_id_2"]) if t1 in target_map and t2 in target_map]
    _add_edges(edge_dict, "target_to_target", [a for a, _ in valid], [b for _, b in valid])
    _add_edges(edge_dict, "target_to_target_rev", [b for a, b in valid], [a for a, b in valid])

    at = tables["adr_target"]
    valid = [(offsets["adr"] + adr_map[a], offsets["target"] + target_map[t]) for a, t in zip(at["Adr_id"], at["Ta_id"]) if a in adr_map and t in target_map]
    _add_edges(edge_dict, "adr_to_target", [a for a, _ in valid], [b for _, b in valid])
    _add_edges(edge_dict, "target_to_adr", [b for a, b in valid], [a for a, b in valid])

    if include_formula:
        formula_map = type_maps["formula"]
        fh = tables["formula_herb"]
        valid = [(offsets["formula"] + formula_map[f], offsets["herb"] + herb_map[h]) for f, h in zip(fh["TCMF_id"], fh["CMM_id"]) if f in formula_map and h in herb_map]
        _add_edges(edge_dict, "formula_to_herb", [a for a, _ in valid], [b for _, b in valid])
        _add_edges(edge_dict, "herb_to_formula", [b for a, b in valid], [a for a, b in valid])
        if split is not None:
            split_df = split["df"].iloc[split["train_idx"]]
            train_pos = split_df[split_df["label"] == 1][["TCMF_id", "Adr_id"]].drop_duplicates()
            valid = [
                (offsets["formula"] + formula_map[f], offsets["adr"] + adr_map[a])
                for f, a in zip(train_pos["TCMF_id"], train_pos["Adr_id"])
                if f in formula_map and a in adr_map
            ]
            _add_edges(edge_dict, "formula_to_adr_train", [a for a, _ in valid], [b for _, b in valid])
            _add_edges(edge_dict, "adr_to_formula_train", [b for a, b in valid], [a for a, b in valid])

    relation_names = sorted(edge_dict.keys())
    rel_to_id = {r: i for i, r in enumerate(relation_names)}
    src_all, dst_all, etype_all = [], [], []
    for rel, (src_list, dst_list) in edge_dict.items():
        src_all.extend(src_list)
        dst_all.extend(dst_list)
        etype_all.extend([rel_to_id[rel]] * len(src_list))

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long, device=DEVICE)
    edge_type = torch.tensor(etype_all, dtype=torch.long, device=DEVICE)
    return {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "total_nodes": total_nodes,
        "offsets": offsets,
        "type_maps": type_maps,
        "relation_names": relation_names,
        "formula_lookup": build_formula_lookup(tables, herb_map),
    }


def build_hgt_graph(tables, type_maps):
    data = HeteroData()
    for node_type, idx_map in type_maps.items():
        data[node_type].num_nodes = len(idx_map)

    hf = tables["herb_ingredient"]
    valid = [(type_maps["herb"][h], type_maps["ingredient"][i]) for h, i in zip(hf["CMM_id"], hf["In_id"]) if h in type_maps["herb"] and i in type_maps["ingredient"]]
    data["herb", "to_ingredient", "ingredient"].edge_index = torch.tensor(valid, dtype=torch.long).t().contiguous()
    data["ingredient", "rev_to_herb", "herb"].edge_index = torch.tensor([(b, a) for a, b in valid], dtype=torch.long).t().contiguous()

    it = tables["ingredient_target"]
    valid = [(type_maps["ingredient"][i], type_maps["target"][t]) for i, t in zip(it["In_id"], it["Ta_id"]) if i in type_maps["ingredient"] and t in type_maps["target"]]
    data["ingredient", "to_target", "target"].edge_index = torch.tensor(valid, dtype=torch.long).t().contiguous()
    data["target", "rev_to_ingredient", "ingredient"].edge_index = torch.tensor([(b, a) for a, b in valid], dtype=torch.long).t().contiguous()

    ht = tables["herb_target"]
    valid = [(type_maps["herb"][h], type_maps["target"][t]) for h, t in zip(ht["CMM_id"], ht["Ta_id"]) if h in type_maps["herb"] and t in type_maps["target"]]
    data["herb", "to_target", "target"].edge_index = torch.tensor(valid, dtype=torch.long).t().contiguous()
    data["target", "rev_to_herb", "herb"].edge_index = torch.tensor([(b, a) for a, b in valid], dtype=torch.long).t().contiguous()

    tt = tables["target_target"]
    valid = [(type_maps["target"][t1], type_maps["target"][t2]) for t1, t2 in zip(tt["Ta_id_1"], tt["Ta_id_2"]) if t1 in type_maps["target"] and t2 in type_maps["target"]]
    data["target", "to_target", "target"].edge_index = torch.tensor(valid, dtype=torch.long).t().contiguous()
    data["target", "rev_to_target", "target"].edge_index = torch.tensor([(b, a) for a, b in valid], dtype=torch.long).t().contiguous()

    at = tables["adr_target"]
    valid = [(type_maps["adr"][a], type_maps["target"][t]) for a, t in zip(at["Adr_id"], at["Ta_id"]) if a in type_maps["adr"] and t in type_maps["target"]]
    data["adr", "to_target", "target"].edge_index = torch.tensor(valid, dtype=torch.long).t().contiguous()
    data["target", "rev_to_adr", "adr"].edge_index = torch.tensor([(b, a) for a, b in valid], dtype=torch.long).t().contiguous()

    data = data.to(DEVICE)
    return data


class PairClassifier(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, formula_z, adr_z):
        return self.net(torch.cat([formula_z, adr_z], dim=-1)).squeeze(-1)


class RGCNEncoder(nn.Module):
    def __init__(self, total_nodes, num_relations, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(total_nodes, hidden)
        self.convs = nn.ModuleList([RGCNConv(hidden, hidden, num_relations) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HGTEncoder(nn.Module):
    def __init__(self, num_nodes_by_type, metadata, hidden=64, layers=2, heads=2, dropout=0.2):
        super().__init__()
        self.emb = nn.ModuleDict({ntype: nn.Embedding(num_nodes, hidden) for ntype, num_nodes in num_nodes_by_type.items()})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, data: HeteroData):
        x_dict = {ntype: emb.weight for ntype, emb in self.emb.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict


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


def pool_formula_embedding(formula_id, herb_embeddings, formula_lookup):
    formula_to_herbs, formula_to_weights = formula_lookup
    herb_idx = formula_to_herbs.get(formula_id, [])
    if not herb_idx:
        return torch.zeros((herb_embeddings.shape[-1],), device=herb_embeddings.device)
    weights = torch.tensor(formula_to_weights.get(formula_id, [1.0 / len(herb_idx)] * len(herb_idx)), dtype=torch.float32, device=herb_embeddings.device)
    weights = weights / weights.sum().clamp_min(1e-8)
    return (herb_embeddings[torch.tensor(herb_idx, device=herb_embeddings.device)] * weights.unsqueeze(-1)).sum(dim=0)


def build_pair_batch_embeddings(df, batch_idx, embeddings, graph_ctx, with_formula_node=False, model_type="rgcn"):
    formula_z, adr_z = [], []
    if model_type == "hgt":
        herb_emb = embeddings["herb"]
        adr_emb = embeddings["adr"]
    else:
        offsets = graph_ctx["offsets"]
        type_maps = graph_ctx["type_maps"]
        herb_start = offsets["herb"]
        adr_start = offsets["adr"]
        herb_emb = embeddings[herb_start : herb_start + len(type_maps["herb"])]
        adr_emb = embeddings[adr_start : adr_start + len(type_maps["adr"])]
        formula_emb = None
        if with_formula_node:
            formula_start = offsets["formula"]
            formula_emb = embeddings[formula_start : formula_start + len(type_maps["formula"])]

    formula_lookup = graph_ctx["formula_lookup"]
    type_maps = graph_ctx["type_maps"]
    for idx in batch_idx:
        row = df.iloc[int(idx)]
        if with_formula_node:
            zf = formula_emb[type_maps["formula"][row["TCMF_id"]]]
        else:
            zf = pool_formula_embedding(row["TCMF_id"], herb_emb, formula_lookup)
        za = adr_emb[type_maps["adr"][row["Adr_id"]]]
        formula_z.append(zf)
        adr_z.append(za)
    return torch.stack(formula_z, dim=0), torch.stack(adr_z, dim=0)


def predict_split(model_name, encoder, classifier, graph_data, graph_ctx, df, labels, indices, with_formula_node=False):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        embeddings = encoder(*graph_data) if model_name.startswith("R-GCN") else encoder(graph_data)
        formula_z, adr_z = build_pair_batch_embeddings(
            df,
            indices,
            embeddings,
            graph_ctx,
            with_formula_node=with_formula_node,
            model_type="rgcn" if model_name.startswith("R-GCN") else "hgt",
        )
        logits = classifier(formula_z, adr_z)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels[np.asarray(indices, dtype=int)]
    return np.asarray(y_true, dtype=int), np.asarray(probs, dtype=float)


def train_one_split(model_name, ds, split, config: GraphConfig, save_result: bool = True):
    set_seed(config.seed + int(split.get("fold", split.get("seed", 0))))
    tables = load_tables()
    with_formula_node = model_name == "R-GCN (w/ Formula)"
    type_maps = make_type_maps(tables, include_formula=with_formula_node)
    labels = ds["df"]["label"].values.astype(int)
    split_local = dict(split)
    split_local["df"] = ds["df"]

    if model_name.startswith("R-GCN"):
        graph_ctx = build_rgcn_graph(tables, type_maps, split=split_local, include_formula=with_formula_node)
        encoder = RGCNEncoder(
            total_nodes=graph_ctx["total_nodes"],
            num_relations=len(graph_ctx["relation_names"]),
            hidden=config.hidden,
            layers=config.layers,
            dropout=config.dropout,
        ).to(DEVICE)
        graph_data = (graph_ctx["edge_index"], graph_ctx["edge_type"])
    else:
        graph_data = build_hgt_graph(tables, type_maps)
        graph_ctx = {
            "type_maps": type_maps,
            "formula_lookup": build_formula_lookup(tables, type_maps["herb"]),
        }
        encoder = HGTEncoder(
            num_nodes_by_type={k: len(v) for k, v in type_maps.items()},
            metadata=graph_data.metadata(),
            hidden=config.hidden,
            layers=config.layers,
            heads=config.heads,
            dropout=config.dropout,
        ).to(DEVICE)

    classifier = PairClassifier(config.hidden, config.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr, weight_decay=config.wd)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)

    best_state = None
    best_val = -np.inf
    bad_rounds = 0
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    train_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    import time

    wall_start = time.time()
    for epoch in range(config.epochs):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        embeddings = encoder(*graph_data) if model_name.startswith("R-GCN") else encoder(graph_data)
        balanced = balance_train_indices(labels, train_idx, neg_ratio=config.neg_ratio, seed=config.seed + epoch)
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(balanced)
        batch_losses = []
        for start_idx in range(0, len(balanced), config.batch_size):
            batch_idx = balanced[start_idx : start_idx + config.batch_size]
            formula_z, adr_z = build_pair_batch_embeddings(
                ds["df"],
                batch_idx,
                embeddings,
                graph_ctx,
                with_formula_node=with_formula_node,
                model_type="rgcn" if model_name.startswith("R-GCN") else "hgt",
            )
            logits = classifier(formula_z, adr_z)
            target = torch.tensor(labels[batch_idx], dtype=torch.float32, device=DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            batch_losses.append(loss)
        if batch_losses:
            torch.stack(batch_losses).mean().backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % config.eval_every == 0:
            y_true_val, y_prob_val = predict_split(
                model_name, encoder, classifier, graph_data, graph_ctx, ds["df"], labels, val_idx, with_formula_node=with_formula_node
            )
            val_metrics = compute_metrics(y_true_val, y_prob_val)
            vscore = val_score_from_metrics(val_metrics)
            if vscore > best_val:
                best_val = vscore
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
                }
                bad_rounds = 0
            else:
                bad_rounds += 1
            if bad_rounds >= max(1, config.patience // config.eval_every):
                break

    train_time_sec = time.time() - wall_start
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        classifier.load_state_dict(best_state["classifier"])

    y_true_val, y_prob_val = predict_split(
        model_name, encoder, classifier, graph_data, graph_ctx, ds["df"], labels, val_idx, with_formula_node=with_formula_node
    )
    val_metrics = compute_metrics(y_true_val, y_prob_val)
    y_true_test, y_prob_test = predict_split(
        model_name, encoder, classifier, graph_data, graph_ctx, ds["df"], labels, test_idx, with_formula_node=with_formula_node
    )
    test_metrics = compute_metrics(y_true_test, y_prob_test, threshold=val_metrics["threshold"])

    fold_id = int(split.get("fold", split.get("seed", 0)))
    state_path = model_state_path(model_name, fold_id)
    if save_result:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}, state_path)

    result = {
        "model": model_name,
        "fold": fold_id,
        "split_key": "fold" if "fold" in split else "seed",
        "y_true": y_true_test,
        "y_prob": y_prob_test,
        "test_indices": test_idx,
        "y_true_val": y_true_val,
        "y_prob_val": y_prob_val,
        "val_indices": val_idx,
        "val_auroc": float(val_metrics["auroc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "threshold": float(val_metrics["threshold"]),
        **{k: float(v) for k, v in test_metrics.items()},
        "train_time_sec": float(train_time_sec),
        "inference_time_sec": 0.0,
        "peak_memory_mb": float(compute_peak_memory_mb()),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "config": config.__dict__.copy(),
        "model_state_path": str(state_path) if save_result else None,
    }
    if save_result:
        save_pickle(result, fold_result_path(model_name, fold_id))
    return result


def run_cv(model_name, ds, fold_splits, config: GraphConfig, fold_subset: list[int] | None = None, save_result: bool = True):
    results = []
    for split in fold_splits:
        split_id = int(split.get("fold", split.get("seed", 0)))
        if fold_subset is not None and split_id not in fold_subset:
            continue
        if model_name == "R-GCN (w/ Formula)" and split.get("group_col") == "TCMF_id":
            continue
        results.append(train_one_split(model_name, ds, split, config, save_result=save_result))
    return results


def summarize_results(results):
    out = {}
    for metric in ["auroc", "auprc", "precision", "recall", "f1", "mcc"]:
        vals = np.asarray([float(r[metric]) for r in results], dtype=float)
        out[f"{metric}_mean"] = float(vals.mean())
        out[f"{metric}_std"] = float(vals.std(ddof=0))
    return out
