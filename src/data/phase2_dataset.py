"""
Clean dataset construction for the Formula-ADR benchmark.

This module builds the tested-universe dataset, feature table, main CV splits,
and seeded cold-start splits used by all later phases.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from experiment_utils import OUT_DIR, ensure_output_dirs, load_pickle, save_pickle


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "final_data_clean"
DATASET_PATH = OUT_DIR / "dataset.pkl"
OUTER_CV_SEED = 42
OUTER_FOLDS = 10
INNER_VAL_FRAC = 1.0 / 9.0
COLD_START_HOLDOUT_FRAC = 0.2
COLD_START_SEEDS = [42, 88, 999, 777, 666]


def load_phase1():
    def _load(name: str):
        return load_pickle(OUT_DIR / name)

    return (
        _load("meta.pkl"),
        _load("herb_target_profiles.pkl"),
        _load("adr_target_profiles.pkl"),
        _load("pathway_tissue_maps.pkl"),
        _load("herb_pair_features.pkl"),
        _load("lookups.pkl"),
    )


def _gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * arr).sum() / (n * arr.sum()))


def build_feature_vector(
    f_id,
    a_id,
    f2h,
    dose_map,
    herb_profiles,
    adr_profiles,
    pair_features,
):
    herbs = f2h.get(f_id, [])
    if not herbs:
        return None

    ap = adr_profiles.get(a_id)
    if ap is None:
        return None

    n_adr_targets = int((ap > 0).sum())

    herb_overlaps = []
    herb_doses = []
    for h in herbs:
        hp = herb_profiles.get(h)
        dr = dose_map.get((f_id, h), {}).get("dose_ratio", 1.0 / len(herbs))
        herb_doses.append(dr)
        if hp is not None:
            overlap = float(np.dot(hp, ap))
            herb_overlaps.append(overlap * dr)
        else:
            herb_overlaps.append(0.0)

    herb_overlaps = np.asarray(herb_overlaps, dtype=float)
    herb_doses = np.asarray(herb_doses, dtype=float)
    n_herbs = len(herbs)

    indiv = {
        "n_herbs": n_herbs,
        "max_indiv_overlap": float(herb_overlaps.max()) if len(herb_overlaps) else 0.0,
        "mean_indiv_overlap": float(herb_overlaps.mean()) if len(herb_overlaps) else 0.0,
        "sum_indiv_overlap": float(herb_overlaps.sum()),
        "std_indiv_overlap": float(herb_overlaps.std()) if n_herbs > 1 else 0.0,
        "frac_herbs_with_overlap": float((herb_overlaps > 0).mean()) if len(herb_overlaps) else 0.0,
        "max_dose_overlap_herb": float(herb_doses[herb_overlaps.argmax()]) if len(herb_overlaps) else 0.0,
        "n_adr_targets": n_adr_targets,
    }

    pair_convergences = []
    pair_complementarities = []
    pair_ppi_bridges = []
    pair_pw_conv = []
    pair_pw_comp = []
    pair_tissue = []
    pair_union = []
    dose_weighted_conv = []

    for h1, h2 in combinations(sorted(herbs), 2):
        pf = pair_features.get((h1, h2, a_id))
        if pf is None:
            continue

        dr1 = dose_map.get((f_id, h1), {}).get("dose_ratio", 1.0 / n_herbs)
        dr2 = dose_map.get((f_id, h2), {}).get("dose_ratio", 1.0 / n_herbs)
        dose_product = dr1 * dr2

        pair_convergences.append(pf["convergence"])
        pair_complementarities.append(pf["complementarity"])
        pair_ppi_bridges.append(pf["ppi_bridge"])
        pair_pw_conv.append(pf["pw_convergence"])
        pair_pw_comp.append(pf["pw_complementarity"])
        pair_tissue.append(pf["tissue_coexpr"])
        pair_union.append(pf["union_coverage"])
        dose_weighted_conv.append(pf["convergence"] * dose_product)

    def _agg(arr):
        if not arr:
            return 0.0, 0.0, 0.0
        arr = np.asarray(arr, dtype=float)
        return float(arr.max()), float(arr.mean()), float(arr.sum())

    pc_max, pc_mean, _ = _agg(pair_convergences)
    pcomp_max, pcomp_mean, _ = _agg(pair_complementarities)
    pppi_max, pppi_mean, pppi_sum = _agg(pair_ppi_bridges)
    pwc_max, pwc_mean, _ = _agg(pair_pw_conv)
    pwcomp_max, pwcomp_mean, _ = _agg(pair_pw_comp)
    tis_max, tis_mean, _ = _agg(pair_tissue)
    un_max, un_mean, _ = _agg(pair_union)
    dwc_max, dwc_mean, dwc_sum = _agg(dose_weighted_conv)

    n_pairs = max(len(pair_convergences), 1)
    frac_conv_pos = sum(1 for c in pair_convergences if c > 0) / n_pairs
    n_high_conv = sum(1 for c in pair_convergences if c > 0.1)

    inter = {
        "max_convergence": pc_max,
        "mean_convergence": pc_mean,
        "max_complementarity": pcomp_max,
        "mean_complementarity": pcomp_mean,
        "max_ppi_bridge": pppi_max,
        "mean_ppi_bridge": pppi_mean,
        "sum_ppi_bridge": pppi_sum,
        "max_pw_convergence": pwc_max,
        "mean_pw_convergence": pwc_mean,
        "max_pw_complementarity": pwcomp_max,
        "mean_pw_complementarity": pwcomp_mean,
        "max_tissue_coexpr": tis_max,
        "mean_tissue_coexpr": tis_mean,
        "max_union_coverage": un_max,
        "mean_union_coverage": un_mean,
        "max_dose_convergence": dwc_max,
        "mean_dose_convergence": dwc_mean,
        "sum_dose_convergence": dwc_sum,
        "frac_convergent_pairs": float(frac_conv_pos),
        "n_high_conv_pairs": float(n_high_conv),
    }

    formula_feats = {
        "total_dose": float(herb_doses.sum()),
        "dose_std": float(herb_doses.std()) if n_herbs > 1 else 0.0,
        "dose_gini": _gini(herb_doses) if n_herbs > 1 else 0.0,
        "n_formula_pairs": float(n_herbs * (n_herbs - 1) / 2),
    }

    return {**indiv, **inter, **formula_feats}


def build_tested_universe():
    signal = pd.read_csv(DATA_DIR / "TCMF_ADR_signal_merged.csv")
    jader_all = pd.read_csv(DATA_DIR / "TCMF_ADR.csv")
    faers_all = pd.read_csv(DATA_DIR / "signal" / "faers_kampo_disproportionality_all.csv")
    tcmf_nodes = pd.read_csv(DATA_DIR / "TCMF_nodes.csv")

    name_to_tcmf = dict(zip(tcmf_nodes["formula_name_jp"], tcmf_nodes["TCMF_id"]))
    faers_all["TCMF_id"] = faers_all["formula_name_jp"].map(name_to_tcmf)
    faers_missing = int(faers_all["TCMF_id"].isna().sum())
    faers_all = faers_all.dropna(subset=["TCMF_id"]).copy()

    jader_pairs = jader_all[["TCMF_id", "Adr_id"]].drop_duplicates().copy()
    jader_pairs["tested_jader"] = True
    jader_pairs["tested_faers"] = False

    faers_pairs = faers_all[["TCMF_id", "Adr_id"]].drop_duplicates().copy()
    faers_pairs["tested_jader"] = False
    faers_pairs["tested_faers"] = True

    tested = pd.concat([jader_pairs, faers_pairs], ignore_index=True)
    tested = (
        tested.groupby(["TCMF_id", "Adr_id"], as_index=False)[["tested_jader", "tested_faers"]]
        .max()
        .sort_values(["TCMF_id", "Adr_id"])
        .reset_index(drop=True)
    )
    tested["tested_source"] = np.where(
        tested["tested_jader"] & tested["tested_faers"],
        "both",
        np.where(tested["tested_jader"], "JADER", "FAERS"),
    )

    signal_meta = signal[["TCMF_id", "Adr_id", "source"]].drop_duplicates().rename(columns={"source": "signal_source"})
    tested = tested.merge(signal_meta, on=["TCMF_id", "Adr_id"], how="left")
    tested["label"] = tested["signal_source"].notna().astype(int)
    tested["signal_source"] = tested["signal_source"].fillna("negative")

    pair_counts = tested.groupby("TCMF_id").size().rename("tested_pair_count").reset_index()
    tested = tested.merge(pair_counts, on="TCMF_id", how="left")

    summary = {
        "n_positive_pairs": int(tested["label"].sum()),
        "n_negative_pairs": int((tested["label"] == 0).sum()),
        "n_tested_pairs": int(len(tested)),
        "n_tested_formulas": int(tested["TCMF_id"].nunique()),
        "n_tested_adrs": int(tested["Adr_id"].nunique()),
        "faers_missing_tcmf_mappings": faers_missing,
    }
    return tested, summary


def compute_split_sanity(df: pd.DataFrame, train_idx, val_idx, test_idx):
    train_pairs = set(zip(df.iloc[train_idx]["TCMF_id"], df.iloc[train_idx]["Adr_id"]))
    val_pairs = set(zip(df.iloc[val_idx]["TCMF_id"], df.iloc[val_idx]["Adr_id"]))
    test_pairs = set(zip(df.iloc[test_idx]["TCMF_id"], df.iloc[test_idx]["Adr_id"]))
    return {
        "train_val_pair_overlap": int(len(train_pairs & val_pairs)),
        "train_test_pair_overlap": int(len(train_pairs & test_pairs)),
        "val_test_pair_overlap": int(len(val_pairs & test_pairs)),
        "n_pos_train": int(df.iloc[train_idx]["label"].sum()),
        "n_pos_val": int(df.iloc[val_idx]["label"].sum()),
        "n_pos_test": int(df.iloc[test_idx]["label"].sum()),
        "n_neg_train": int((df.iloc[train_idx]["label"] == 0).sum()),
        "n_neg_val": int((df.iloc[val_idx]["label"] == 0).sum()),
        "n_neg_test": int((df.iloc[test_idx]["label"] == 0).sum()),
    }


def build_fold_splits(df: pd.DataFrame):
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=OUTER_CV_SEED)
    fold_splits = []
    for fold_id, (train_val_idx, test_idx) in enumerate(outer.split(df.index.values, df["label"].values)):
        train_val_idx = np.asarray(train_val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_FRAC,
            random_state=OUTER_CV_SEED + fold_id,
        )
        inner_train, inner_val = next(inner.split(train_val_idx, df.iloc[train_val_idx]["label"].values))
        train_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]

        fold_splits.append(
            {
                "fold": fold_id,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "sanity": compute_split_sanity(df, train_idx, val_idx, test_idx),
            }
        )
    return fold_splits


def compute_group_holdout_sanity(df: pd.DataFrame, group_col: str, train_idx, val_idx, test_idx):
    train_groups = set(df.iloc[train_idx][group_col])
    val_groups = set(df.iloc[val_idx][group_col])
    test_groups = set(df.iloc[test_idx][group_col])
    return {
        "train_val_group_overlap": int(len(train_groups & val_groups)),
        "train_test_group_overlap": int(len(train_groups & test_groups)),
        "val_test_group_overlap": int(len(val_groups & test_groups)),
        "n_pos_train": int(df.iloc[train_idx]["label"].sum()),
        "n_pos_val": int(df.iloc[val_idx]["label"].sum()),
        "n_pos_test": int(df.iloc[test_idx]["label"].sum()),
    }


def build_group_holdout_splits(df: pd.DataFrame, group_col: str, seeds: list[int], holdout_frac: float):
    positive_groups = sorted(df.loc[df["label"] == 1, group_col].unique().tolist())
    splits = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        groups = positive_groups.copy()
        rng.shuffle(groups)
        n_holdout = max(1, int(round(len(groups) * holdout_frac)))
        holdout_groups = set(groups[:n_holdout])

        test_mask = df[group_col].isin(holdout_groups).values
        train_val_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        inner_train, inner_val = next(inner.split(train_val_idx, df.iloc[train_val_idx]["label"].values))
        train_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]

        splits.append(
            {
                "seed": seed,
                "group_col": group_col,
                "holdout_groups": sorted(holdout_groups),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "sanity": compute_group_holdout_sanity(df, group_col, train_idx, val_idx, test_idx),
            }
        )
    return splits


def build_dataset():
    ensure_output_dirs()

    print("Phase 2: Feature engineering and dataset construction")
    print("=" * 60)

    print("\n[1/5] Loading Phase 1 outputs and raw tables ...")
    _, herb_profiles, adr_profiles, _, pair_features, lookups = load_phase1()
    f2h = lookups["f2h"]
    dose_map = lookups["dose_map"]

    print("\n[2/5] Building tested-universe pair table ...")
    tested_df, tested_summary = build_tested_universe()
    print(
        f"  Tested pairs: {tested_summary['n_tested_pairs']} "
        f"({tested_summary['n_positive_pairs']} pos, {tested_summary['n_negative_pairs']} neg)"
    )
    print(f"  Tested formulas: {tested_summary['n_tested_formulas']}, tested ADRs: {tested_summary['n_tested_adrs']}")
    if tested_summary["faers_missing_tcmf_mappings"] > 0:
        print(f"  Warning: {tested_summary['faers_missing_tcmf_mappings']} FAERS rows lacked TCMF mapping and were dropped.")

    print("\n[3/5] Building per-pair feature vectors ...")
    records = []
    missing_pairs = 0
    for _, row in tested_df.iterrows():
        fv = build_feature_vector(
            row["TCMF_id"],
            row["Adr_id"],
            f2h,
            dose_map,
            herb_profiles,
            adr_profiles,
            pair_features,
        )
        if fv is None:
            missing_pairs += 1
            continue
        record = dict(row)
        record.update(fv)
        records.append(record)

    df = pd.DataFrame(records).sort_values(["TCMF_id", "Adr_id"]).reset_index(drop=True)
    df["sample_id"] = np.arange(len(df), dtype=int)
    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "sample_id",
            "TCMF_id",
            "Adr_id",
            "label",
            "tested_jader",
            "tested_faers",
            "tested_source",
            "signal_source",
            "tested_pair_count",
        }
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    print(f"  Feature rows built: {len(df)} (skipped {missing_pairs})")
    print(f"  Feature dimensions: {len(feature_cols)}")

    print("\n[4/5] Building main CV splits ...")
    fold_splits = build_fold_splits(df)
    for split in fold_splits[:2]:
        sanity = split["sanity"]
        print(
            f"  Fold {split['fold']}: train={split['n_train']} val={split['n_val']} test={split['n_test']} "
            f"| overlaps={sanity['train_val_pair_overlap']}/{sanity['train_test_pair_overlap']}/{sanity['val_test_pair_overlap']}"
        )

    print("\n[5/5] Building seeded cold-start splits ...")
    formula_cs_splits = build_group_holdout_splits(df, "TCMF_id", COLD_START_SEEDS, COLD_START_HOLDOUT_FRAC)
    adr_cs_splits = build_group_holdout_splits(df, "Adr_id", COLD_START_SEEDS, COLD_START_HOLDOUT_FRAC)
    print(f"  Formula cold-start seeds: {[s['seed'] for s in formula_cs_splits]}")
    print(f"  ADR cold-start seeds: {[s['seed'] for s in adr_cs_splits]}")

    dataset = {
        "df": df,
        "feature_cols": feature_cols,
        "fold_splits": fold_splits,
        "formula_cs_splits": formula_cs_splits,
        "adr_cs_splits": adr_cs_splits,
        "tested_universe_summary": tested_summary,
        "config": {
            "outer_folds": OUTER_FOLDS,
            "outer_seed": OUTER_CV_SEED,
            "inner_val_frac": INNER_VAL_FRAC,
            "cold_start_holdout_frac": COLD_START_HOLDOUT_FRAC,
            "cold_start_seeds": COLD_START_SEEDS,
        },
    }
    save_pickle(dataset, DATASET_PATH)

    print(f"\nDataset saved to {DATASET_PATH}")
    print("Phase 2 complete.")


if __name__ == "__main__":
    build_dataset()
