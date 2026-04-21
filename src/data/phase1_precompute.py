"""Pre-compute herb, ingredient, ADR, and herb-pair profiles from the KG.

Outputs (written to ``../outputs/``):

* ``herb_target_profiles.pkl``       -- {CMM_id: vector over targets}
* ``ingredient_target_profiles.pkl`` -- {In_id: vector over targets}
* ``adr_target_profiles.pkl``        -- {Adr_id: vector over targets}
* ``kg_profile_embeddings.pkl``      -- {ingredient/herb/adr: KG-aware vector}
* ``target_pathway_map.pkl``         -- {Ta_id: set of pathway_ids}
* ``target_tissue_map.pkl``          -- {Ta_id: set of tissue names}
* ``herb_pair_features.pkl``         -- {(CMM_id1, CMM_id2, Adr_id): features}
* ``ingredient_pair_index.pkl``      -- traceability index for interpretability
"""

import os, pickle, sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "final_data_clean")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_tables():
    d = DATA_DIR
    tables = {
        "formula_herb": pd.read_csv(f"{d}/CMM_TCMF.csv"),
        "herb_ingredient": pd.read_csv(f"{d}/CMM_Ingredient.csv"),
        "ingredient_target": pd.read_csv(f"{d}/Ingredient_Target_.csv"),
        "herb_target": pd.read_csv(f"{d}/CMM_Target.csv"),
        "target_target": pd.read_csv(f"{d}/Target_Target.csv"),
        "adr_target": pd.read_csv(f"{d}/ADR_Target.csv"),
        "signal": pd.read_csv(f"{d}/TCMF_ADR_signal_merged.csv"),
        "ingredient_nodes": pd.read_csv(f"{d}/Ingredient_nodes.csv"),
        "adr_nodes": pd.read_csv(f"{d}/ADR_nodes.csv"),
        "cmm_nodes": pd.read_csv(f"{d}/CMM_nodes.csv"),
    }
    mm = f"{d}/multiomics"
    tables["ta_pathway"] = pd.read_csv(f"{mm}/kg_edges_ta_reactome.tsv", sep="\t")
    tables["ta_tissue"] = pd.read_csv(f"{mm}/kg_edges_ta_tissue_rna.tsv", sep="\t")
    return tables


def build_lookups(tables):
    h2i = tables["herb_ingredient"].groupby("CMM_id")["In_id"].apply(set).to_dict()
    i2t = tables["ingredient_target"].groupby("In_id")["Ta_id"].apply(set).to_dict()
    h2t_direct = tables["herb_target"].groupby("CMM_id")["Ta_id"].apply(set).to_dict()
    a2t = tables["adr_target"].groupby("Adr_id")["Ta_id"].apply(set).to_dict()

    ppi = defaultdict(set)
    for _, r in tables["target_target"].iterrows():
        ppi[r["Ta_id_1"]].add(r["Ta_id_2"])
        ppi[r["Ta_id_2"]].add(r["Ta_id_1"])

    f2h = tables["formula_herb"].groupby("TCMF_id")["CMM_id"].apply(list).to_dict()

    dose_map = {}
    for _, r in tables["formula_herb"].iterrows():
        dose_map[(r["TCMF_id"], r["CMM_id"])] = {
            "dose_g": r["dose_g"],
            "dose_ratio": r["dose_ratio"],
        }

    return dict(
        h2i=h2i, i2t=i2t, h2t_direct=h2t_direct, a2t=a2t,
        ppi=ppi, f2h=f2h, dose_map=dose_map,
    )


# ── 1.1  Herb target profiles ──────────────────────────────────────────────

def build_herb_target_profiles(lookups, all_targets_list):
    """Binary vector per herb: 1 if herb reaches that target."""
    h2i = lookups["h2i"]
    i2t = lookups["i2t"]
    h2t_direct = lookups["h2t_direct"]
    tidx = {t: i for i, t in enumerate(all_targets_list)}
    n = len(all_targets_list)

    profiles = {}
    all_herbs = set(h2i.keys()) | set(h2t_direct.keys())
    for h in all_herbs:
        targets = set()
        if h in h2t_direct:
            targets |= h2t_direct[h]
        if h in h2i:
            for ing in h2i[h]:
                targets |= i2t.get(ing, set())
        vec = np.zeros(n, dtype=np.float32)
        for t in targets:
            if t in tidx:
                vec[tidx[t]] = 1.0
        profiles[h] = vec
    return profiles


def build_ingredient_target_profiles(lookups, all_targets_list):
    """Binary vector per ingredient: 1 if ingredient reaches that target."""
    i2t = lookups["i2t"]
    tidx = {t: i for i, t in enumerate(all_targets_list)}
    n = len(all_targets_list)

    profiles = {}
    for ing, targets in i2t.items():
        vec = np.zeros(n, dtype=np.float32)
        for t in targets:
            if t in tidx:
                vec[tidx[t]] = 1.0
        profiles[ing] = vec
    return profiles


# ── 1.2  ADR target profiles ───────────────────────────────────────────────

def build_adr_target_profiles(lookups, all_targets_list, tables):
    """Binary vector per ADR. For ADRs without targets, attempt MedDRA hierarchy propagation."""
    a2t = lookups["a2t"]
    tidx = {t: i for i, t in enumerate(all_targets_list)}
    n = len(all_targets_list)

    adr_nodes = tables["adr_nodes"]

    meddra_to_adr = {}
    adr_to_meddra = {}
    if "MedDRA_ID" in adr_nodes.columns:
        for _, r in adr_nodes.iterrows():
            mid = r.get("MedDRA_ID")
            if pd.notna(mid):
                meddra_to_adr[mid] = r["Adr_id"]
                adr_to_meddra[r["Adr_id"]] = mid

    profiles = {}
    no_target_count = 0
    for _, r in adr_nodes.iterrows():
        a = r["Adr_id"]
        targets = a2t.get(a, set())
        if not targets:
            no_target_count += 1
        vec = np.zeros(n, dtype=np.float32)
        for t in targets:
            if t in tidx:
                vec[tidx[t]] = 1.0
        profiles[a] = vec

    print(f"  ADR profiles built: {len(profiles)} total, {no_target_count} without direct targets")
    return profiles


# ── 1.3  Pathway / tissue mappings ─────────────────────────────────────────

def build_pathway_tissue_maps(tables):
    ta_pw = tables["ta_pathway"]
    t2pw = ta_pw.groupby("Ta_id")["reactome_id"].apply(set).to_dict()

    ta_ti = tables["ta_tissue"]
    tissue_col = [c for c in ta_ti.columns if c not in ["Ta_id", "uniprot"]][0]
    t2tissue = ta_ti.groupby("Ta_id")[tissue_col].apply(set).to_dict()

    pw2t = ta_pw.groupby("reactome_id")["Ta_id"].apply(set).to_dict()

    return t2pw, t2tissue, pw2t


def build_target_transition_matrix(tables, all_targets_list):
    tidx = {t: i for i, t in enumerate(all_targets_list)}
    rows = []
    cols = []
    for _, r in tables["target_target"].iterrows():
        t1 = r["Ta_id_1"]
        t2 = r["Ta_id_2"]
        if t1 not in tidx or t2 not in tidx:
            continue
        i = tidx[t1]
        j = tidx[t2]
        rows.extend([i, j])
        cols.extend([j, i])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(len(all_targets_list), len(all_targets_list)), dtype=np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    deg[deg == 0] = 1.0
    return sparse.diags(1.0 / deg) @ adj


def diffuse_profile(vec, transition, alpha=0.35, steps=2):
    base = np.asarray(vec, dtype=np.float32).ravel()
    out = base.copy()
    current = base.copy()
    for step in range(steps):
        current = np.asarray(transition.dot(current)).ravel().astype(np.float32)
        out += (alpha ** (step + 1)) * current
    return out.astype(np.float32)


def build_static_embedding_bundle(ingredient_profiles, herb_profiles, adr_profiles, transition, dim=48):
    ing_ids = sorted(ingredient_profiles)
    herb_ids = sorted(herb_profiles)
    adr_ids = sorted(adr_profiles)

    stack = []
    for ing in ing_ids:
        stack.append(diffuse_profile(ingredient_profiles[ing], transition))
    for herb in herb_ids:
        stack.append(diffuse_profile(herb_profiles[herb], transition))
    for adr in adr_ids:
        stack.append(diffuse_profile(adr_profiles[adr], transition))
    stack = np.stack(stack).astype(np.float32)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    reduced = svd.fit_transform(stack).astype(np.float32)

    cursor = 0
    ingredient_red = {ing: reduced[cursor + i] for i, ing in enumerate(ing_ids)}
    cursor += len(ing_ids)
    herb_red = {herb: reduced[cursor + i] for i, herb in enumerate(herb_ids)}
    cursor += len(herb_ids)
    adr_red = {adr: reduced[cursor + i] for i, adr in enumerate(adr_ids)}
    return {"dim": dim, "ingredient": ingredient_red, "herb": herb_red, "adr": adr_red}


# ── 1.4  Pairwise herb interaction features ────────────────────────────────

def compute_pair_features(
    h1, h2, adr_id,
    herb_profiles, adr_profiles, lookups,
    t2pw, t2tissue, pw2t, all_targets_list
):
    hp1 = herb_profiles.get(h1)
    hp2 = herb_profiles.get(h2)
    ap = adr_profiles.get(adr_id)
    if hp1 is None or hp2 is None or ap is None:
        return None

    tidx = {t: i for i, t in enumerate(all_targets_list)}
    ppi = lookups["ppi"]

    adr_target_set = set(np.where(ap > 0)[0])
    h1_target_set = set(np.where(hp1 > 0)[0])
    h2_target_set = set(np.where(hp2 > 0)[0])

    h1_adr = h1_target_set & adr_target_set
    h2_adr = h2_target_set & adr_target_set
    n_adr = len(adr_target_set)
    if n_adr == 0:
        return _zero_features()

    # Target-level
    convergence_set = h1_adr & h2_adr
    h1_only = h1_adr - h2_adr
    h2_only = h2_adr - h1_adr
    complementarity_set = h1_only | h2_only
    union_set = h1_adr | h2_adr

    convergence = len(convergence_set) / n_adr
    complementarity = len(complementarity_set) / n_adr
    union_coverage = len(union_set) / n_adr

    # PPI bridge: count PPI edges between h1_adr targets and h2_adr targets
    ppi_bridge = 0
    adr_targets_by_idx = {all_targets_list[i] for i in adr_target_set}
    h1_adr_ids = {all_targets_list[i] for i in h1_adr}
    h2_adr_ids = {all_targets_list[i] for i in h2_adr}
    for t1 in h1_adr_ids:
        ppi_bridge += len(ppi.get(t1, set()) & h2_adr_ids)

    # Pathway-level
    h1_pathways = set()
    h2_pathways = set()
    adr_pathways = set()
    for i in h1_adr:
        h1_pathways |= t2pw.get(all_targets_list[i], set())
    for i in h2_adr:
        h2_pathways |= t2pw.get(all_targets_list[i], set())
    for i in adr_target_set:
        adr_pathways |= t2pw.get(all_targets_list[i], set())

    pw_convergence = len(h1_pathways & h2_pathways & adr_pathways)
    pw_complementarity = len((h1_pathways ^ h2_pathways) & adr_pathways)

    # Tissue-level
    h1_tissues = set()
    h2_tissues = set()
    for i in h1_adr:
        h1_tissues |= t2tissue.get(all_targets_list[i], set())
    for i in h2_adr:
        h2_tissues |= t2tissue.get(all_targets_list[i], set())
    tissue_coexpr = len(h1_tissues & h2_tissues)

    return {
        "convergence": convergence,
        "complementarity": complementarity,
        "union_coverage": union_coverage,
        "ppi_bridge": ppi_bridge,
        "pw_convergence": pw_convergence,
        "pw_complementarity": pw_complementarity,
        "tissue_coexpr": tissue_coexpr,
        "n_adr_targets": n_adr,
        "n_h1_adr_targets": len(h1_adr),
        "n_h2_adr_targets": len(h2_adr),
        "n_convergence_targets": len(convergence_set),
    }


def _zero_features():
    return {
        "convergence": 0.0,
        "complementarity": 0.0,
        "union_coverage": 0.0,
        "ppi_bridge": 0,
        "pw_convergence": 0,
        "pw_complementarity": 0,
        "tissue_coexpr": 0,
        "n_adr_targets": 0,
        "n_h1_adr_targets": 0,
        "n_h2_adr_targets": 0,
        "n_convergence_targets": 0,
    }


# ── 1.5  Ingredient-pair traceability index ────────────────────────────────

def build_ingredient_pair_index(h1, h2, adr_id, lookups, adr_profiles, all_targets_list):
    """For interpretability: which ingredient pairs contribute to convergence?"""
    h2i = lookups["h2i"]
    i2t = lookups["i2t"]
    ap = adr_profiles.get(adr_id)
    if ap is None:
        return []

    adr_targets = {all_targets_list[i] for i in np.where(ap > 0)[0]}
    if not adr_targets:
        return []

    ings1 = h2i.get(h1, set())
    ings2 = h2i.get(h2, set())
    records = []
    for i1 in ings1:
        t1 = i2t.get(i1, set())
        t1_adr = t1 & adr_targets
        if not t1_adr:
            continue
        for i2 in ings2:
            t2 = i2t.get(i2, set())
            shared = t1_adr & (t2 & adr_targets)
            if shared:
                records.append((i1, i2, shared))
    return records


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    print("Phase 1: Pre-computing herb interaction profiles from KG")
    print("=" * 60)

    print("\n[1/6] Loading tables ...")
    tables = load_tables()
    lookups = build_lookups(tables)

    all_targets = sorted(
        set(tables["ingredient_target"]["Ta_id"])
        | set(tables["adr_target"]["Ta_id"])
        | set(tables["herb_target"]["Ta_id"])
        | set(tables["target_target"]["Ta_id_1"])
        | set(tables["target_target"]["Ta_id_2"])
    )
    print(f"  Total unique targets: {len(all_targets)}")

    print("\n[2/6] Building herb and ingredient target profiles ...")
    herb_profiles = build_herb_target_profiles(lookups, all_targets)
    ingredient_profiles = build_ingredient_target_profiles(lookups, all_targets)
    print(f"  Built profiles for {len(herb_profiles)} herbs")
    print(f"  Built profiles for {len(ingredient_profiles)} ingredients")

    print("\n[3/6] Building ADR target profiles ...")
    adr_profiles = build_adr_target_profiles(lookups, all_targets, tables)

    print("\n[4/6] Building pathway / tissue mappings ...")
    t2pw, t2tissue, pw2t = build_pathway_tissue_maps(tables)
    target_transition = build_target_transition_matrix(tables, all_targets)
    kg_embeddings = build_static_embedding_bundle(ingredient_profiles, herb_profiles, adr_profiles, target_transition)
    print(f"  Targets with pathway info: {len(t2pw)}")
    print(f"  Targets with tissue info: {len(t2tissue)}")
    print(
        "  Built KG-aware static embeddings: "
        f"ingredient={len(kg_embeddings['ingredient'])}, "
        f"herb={len(kg_embeddings['herb'])}, "
        f"adr={len(kg_embeddings['adr'])}"
    )

    print("\n[5/6] Computing pairwise herb interaction features ...")
    signal = tables["signal"][["TCMF_id", "Adr_id"]].drop_duplicates()
    f2h = lookups["f2h"]

    needed_triplets = set()
    signal_formulas = signal["TCMF_id"].unique()
    signal_adrs = signal["Adr_id"].unique()
    signal_set = set(zip(signal["TCMF_id"], signal["Adr_id"]))

    for f_id in signal_formulas:
        herbs = f2h.get(f_id, [])
        for a_id in signal_adrs:
            for h1, h2 in combinations(sorted(herbs), 2):
                needed_triplets.add((h1, h2, a_id))

    print(f"  Unique (herb1, herb2, ADR) triplets to compute: {len(needed_triplets)}")

    pair_features = {}
    ingredient_index = {}
    done = 0
    for h1, h2, a_id in needed_triplets:
        feat = compute_pair_features(
            h1, h2, a_id,
            herb_profiles, adr_profiles, lookups,
            t2pw, t2tissue, pw2t, all_targets,
        )
        if feat is not None:
            pair_features[(h1, h2, a_id)] = feat

        idx = build_ingredient_pair_index(h1, h2, a_id, lookups, adr_profiles, all_targets)
        if idx:
            ingredient_index[(h1, h2, a_id)] = idx

        done += 1
        if done % 50000 == 0:
            print(f"    ... {done}/{len(needed_triplets)} computed")

    print(f"  Computed features for {len(pair_features)} triplets")
    print(f"  Ingredient traceability entries: {len(ingredient_index)}")

    print("\n[6/6] Saving outputs ...")
    meta = {
        "all_targets": all_targets,
        "signal_formulas": list(signal_formulas),
        "signal_adrs": list(signal_adrs),
    }
    with open(f"{OUT_DIR}/meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(f"{OUT_DIR}/herb_target_profiles.pkl", "wb") as f:
        pickle.dump(herb_profiles, f)
    with open(f"{OUT_DIR}/ingredient_target_profiles.pkl", "wb") as f:
        pickle.dump(ingredient_profiles, f)
    with open(f"{OUT_DIR}/adr_target_profiles.pkl", "wb") as f:
        pickle.dump(adr_profiles, f)
    with open(f"{OUT_DIR}/kg_profile_embeddings.pkl", "wb") as f:
        pickle.dump(kg_embeddings, f)
    with open(f"{OUT_DIR}/pathway_tissue_maps.pkl", "wb") as f:
        pickle.dump({"t2pw": t2pw, "t2tissue": t2tissue, "pw2t": pw2t}, f)
    with open(f"{OUT_DIR}/herb_pair_features.pkl", "wb") as f:
        pickle.dump(pair_features, f)
    with open(f"{OUT_DIR}/ingredient_pair_index.pkl", "wb") as f:
        pickle.dump(ingredient_index, f)
    with open(f"{OUT_DIR}/lookups.pkl", "wb") as f:
        pickle.dump(lookups, f)

    print(f"\nPhase 1 complete. All outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
