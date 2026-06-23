r"""Build an unbiased random sample of high-scoring non-signal predictions.

The previous Supp Table~S9 took the *top* 15 high-scoring non-signal
predictions, which biases the table toward the very easiest cases and
risks systematic cherry-picking. The old-paper protocol (see Kondo
et al.\ 2024 Table 6) is instead:

    "random sample of 15 high-scoring non-signal Kampo formula-ADR
    candidate pairs"

This script reproduces that protocol:

1. Load the canonical 10-fold OOF predictions and join on the full
   dataset (TCMF_id, Adr_id, label, tested_source, herb list).
2. Restrict to label == 0 (non-signal) and prob >= 0.5
   (high-scoring; above the standard classification threshold and
   substantially above the 0.178 base-rate).
3. Randomly sample 15 with a fixed numpy seed of 2024 so the
   sample is reproducible.
4. Recompute top-attended herb and top-scoring pair exactly as
   ``phase5_interpretability.run_top_novel_predictions`` does.

Outputs
-------
- ``results/.../interpretability/random_external_validation_pool.csv``
  (the full label=0 prob>=0.5 pool, for transparency)
- ``results/.../interpretability/random_external_validation_sample.csv``
  (the 15-row random sample)
"""

from __future__ import annotations

import pathlib as _pathlib
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd


ROOT = _pathlib.Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
for p in (SRC, SRC / "models", SRC / "evaluation", SRC / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_PKL = ROOT / "outputs" / "dataset.pkl"
OOF_PKL = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "oof_predictions_with_attention.pkl"
INTERPRET = OOF_PKL.parent
ROMAJI = ROOT / "final_data_clean" / "TCMF_nodes_with_romaji.csv"
HERB_NODES = ROOT / "final_data_clean" / "CMM_nodes.csv"
ADR_NODES = ROOT / "final_data_clean" / "ADR_nodes.csv"
FORMULA_HERB = ROOT / "final_data_clean" / "CMM_TCMF.csv"

PROB_THRESHOLD = 0.50
N_SAMPLE = 15
SEED = 2024


def _row_to_attn_array(v, expected_len: int) -> np.ndarray:
    if v is None:
        return np.zeros(0, dtype=float)
    arr = np.asarray(v, dtype=float).ravel()
    if expected_len > 0 and arr.size != expected_len:
        if arr.size > expected_len:
            arr = arr[:expected_len]
        else:
            arr = np.pad(arr, (0, expected_len - arr.size))
    return arr


def main() -> int:
    with open(DATA_PKL, "rb") as fh:
        ds = pickle.load(fh)
    df = ds["df"].reset_index(drop=True).copy()
    df["sample_idx"] = df.index.astype(int)

    with open(OOF_PKL, "rb") as fh:
        oof = pickle.load(fh)

    romaji = pd.read_csv(ROMAJI)[["TCMF_id", "formula_name_jp", "formula_name_romaji"]]
    herb_nodes = pd.read_csv(HERB_NODES)
    adr_nodes = pd.read_csv(ADR_NODES)
    f_h = pd.read_csv(FORMULA_HERB)

    herb_name_col = [c for c in herb_nodes.columns if "name" in c.lower() and "en" in c.lower()]
    herb_name_col = herb_name_col[0] if herb_name_col else (
        "CMM_name_EN" if "CMM_name_EN" in herb_nodes.columns else herb_nodes.columns[1]
    )
    herb_name = dict(zip(herb_nodes["CMM_id"], herb_nodes[herb_name_col]))

    adr_name_col = [c for c in adr_nodes.columns if "name" in c.lower()]
    adr_name_col = adr_name_col[0] if adr_name_col else (
        "Adr_name" if "Adr_name" in adr_nodes.columns else adr_nodes.columns[1]
    )
    adr_name = dict(zip(adr_nodes["Adr_id"], adr_nodes[adr_name_col]))

    f2h: dict[str, list[str]] = {}
    for fid, grp in f_h.groupby("TCMF_id"):
        f2h[fid] = sorted(grp["CMM_id"].tolist())

    merged = df.merge(oof, on="sample_idx", how="inner")
    merged["label"] = merged["label"].astype(int)
    pool = merged[(merged["label"] == 0) & (merged["prob"] >= PROB_THRESHOLD)].copy()
    pool = pool.sort_values("prob", ascending=False).reset_index(drop=True)
    print(f"[pool] total OOF samples = {len(merged)}", flush=True)
    print(f"[pool] non-signal (label=0) with prob>={PROB_THRESHOLD}: {len(pool)}", flush=True)

    rng = np.random.default_rng(SEED)
    if len(pool) < N_SAMPLE:
        raise ValueError(f"Pool too small ({len(pool)}) to draw {N_SAMPLE}.")
    sample_idx = rng.choice(len(pool), size=N_SAMPLE, replace=False)
    sample_idx = np.sort(sample_idx)  # keep stable print order
    sample = pool.iloc[sample_idx].reset_index(drop=True).copy()

    rows = []
    for _, entry in sample.iterrows():
        f_id = entry["TCMF_id"]
        a_id = entry["Adr_id"]
        herbs = f2h.get(f_id, [])
        alpha = _row_to_attn_array(entry.get("herb_attn"), len(herbs))
        top_herb_idx = int(np.argmax(alpha)) if alpha.size > 0 else -1
        top_herb_id = herbs[top_herb_idx] if 0 <= top_herb_idx < len(herbs) else ""
        pair_attn = _row_to_attn_array(entry.get("pair_attn"), 0)
        expected_pairs = list(combinations(range(len(herbs)), 2))
        if pair_attn.size == len(expected_pairs) and pair_attn.size > 0:
            top_pair_idx = int(np.argmax(pair_attn))
            h1_idx, h2_idx = expected_pairs[top_pair_idx]
            top_pair = f"{herb_name.get(herbs[h1_idx], herbs[h1_idx])} x {herb_name.get(herbs[h2_idx], herbs[h2_idx])}"
        else:
            top_pair = ""
        formula_kanji = romaji[romaji["TCMF_id"] == f_id]["formula_name_jp"].values
        formula_romaji = romaji[romaji["TCMF_id"] == f_id]["formula_name_romaji"].values
        rows.append({
            "TCMF_id": f_id,
            "formula_kanji": formula_kanji[0] if len(formula_kanji) else f_id,
            "formula_romaji": formula_romaji[0] if len(formula_romaji) else f_id,
            "Adr_id": a_id,
            "ADR_name": adr_name.get(a_id, a_id),
            "prob": float(entry["prob"]),
            "fold": int(entry["fold"]) if pd.notna(entry.get("fold")) else -1,
            "top_herb": herb_name.get(top_herb_id, top_herb_id),
            "top_pair": top_pair,
            "tested_source": entry.get("tested_source", ""),
        })
    out_sample = pd.DataFrame(rows)

    out_pool_csv = INTERPRET / "random_external_validation_pool.csv"
    out_sample_csv = INTERPRET / "random_external_validation_sample.csv"
    pool.drop(columns=[c for c in ("herb_attn", "pair_attn") if c in pool.columns]).to_csv(out_pool_csv, index=False)
    out_sample.to_csv(out_sample_csv, index=False)
    print(f"[pool] wrote {out_pool_csv}  ({len(pool)} rows)")
    print(f"[sample] wrote {out_sample_csv}  ({len(out_sample)} rows, seed={SEED})")
    print()
    print(out_sample[["formula_romaji", "formula_kanji", "ADR_name", "prob", "top_herb", "top_pair"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
