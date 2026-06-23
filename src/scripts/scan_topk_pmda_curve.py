"""Scan PMDA hit rate at multiple top-K cutoffs.

For each K in a sensible grid we report:
  - HerbPairIAM top-K hit rate (Yes / K)
  - ADR-matched random hit rate (200 random pairs whose ADR
    distribution matches the top-K)
  - unstratified random hit rate (200 random pairs from the full
    non-signal universe)
  - the resulting enrichment ratios

The audit machinery is identical to ``audit_top200_nonsignal.py``
and ``audit_random_nonsignal_baseline.py`` (same Section-11 keyword
list, same family map, same ``pdftotext`` pipeline). We re-use the
already-computed ``top200_nonsignal_pmda_audit.csv`` as a fast
lookup so the per-K query reads from the cached row labels.
"""

from __future__ import annotations

import pathlib as _pathlib
import pickle
import re
import subprocess
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


ROOT = _pathlib.Path(__file__).resolve().parents[2]
DATASET_PKL = ROOT / "outputs" / "dataset.pkl"
OOF_PKL = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "oof_predictions_with_attention.pkl"
INSERT_DIR = ROOT / "药品说明书"
ROMAJI_CSV = ROOT / "final_data_clean" / "TCMF_nodes_with_romaji.csv"
HERB_NODES = ROOT / "final_data_clean" / "CMM_nodes.csv"
ADR_NODES = ROOT / "final_data_clean" / "ADR_nodes.csv"
FORMULA_HERB = ROOT / "final_data_clean" / "CMM_TCMF.csv"


# ============================================================
# ADR family synonyms (IDENTICAL to audit_top200_nonsignal.py)
# ============================================================

def _hep():
    return {
        "direct": [
            "肝機能障害", "肝機能異常", "肝障害", "肝炎", "黄疸",
            "劇症肝炎", "肝不全", "胆汁うっ滞", "胆石", "胆管炎",
            "腹水",
            "AST", "ALT", "ALP", "γ-GTP", "γGTP", "Al-P",
            "アルカリホスファターゼ", "ビリルビン", "トランスアミナーゼ",
        ],
        "symptom": [],
    }

def _ild():
    return {"direct": ["間質性肺炎", "間質性肺疾患", "肺臓炎", "肺線維症",
                       "肺胞炎", "肺炎", "肺胞出血", "呼吸困難"], "symptom": []}

def _hpk():
    return {"direct": ["低カリウム血症", "低K血症", "偽アルドステロン症",
                       "ミオパチー", "横紋筋融解症"], "symptom": []}

def _skn():
    return {"direct": ["発疹", "発赤", "蕁麻疹", "皮疹", "薬疹",
                       "中毒性皮疹", "中毒性表皮壊死症", "TEN",
                       "皮膚粘膜眼症候群", "Stevens-Johnson",
                       "スチーブンス・ジョンソン", "重症薬疹",
                       "湿疹", "固定薬疹", "膿疱"], "symptom": ["瘙痒"]}

def _dig():
    return {"direct": ["食欲不振", "悪心", "嘔吐", "下痢", "腹痛", "便秘",
                       "胃部不快感", "腸閉塞", "イレウス"], "symptom": []}

def _ns():
    return {"direct": ["めまい", "頭痛", "不眠", "眠気", "倦怠感",
                       "しびれ", "ふらつき", "末梢神経障害", "せん妄",
                       "意識障害", "麻痺", "不穏", "興奮", "幻覚"], "symptom": []}

def _card():
    return {"direct": ["動悸", "頻脈", "心悸亢進", "不整脈", "血圧上昇",
                       "高血圧", "低血圧", "浮腫", "心筋梗塞", "心不全"], "symptom": []}

def _ele():
    return {"direct": ["低ナトリウム血症", "高ナトリウム血症",
                       "低カルシウム血症", "高カルシウム血症",
                       "電解質異常"], "symptom": []}

def _bld():
    return {"direct": ["血小板減少", "貧血", "白血球減少", "顆粒球減少",
                       "汎血球減少", "血小板数"], "symptom": []}

def _uro():
    return {"direct": ["尿閉", "排尿障害", "腎機能障害", "腎不全",
                       "間質性腎炎", "尿細管間質性腎炎", "クレアチニン",
                       "BUN"], "symptom": []}

def _sys_chills(): return {"direct": ["悪寒"], "symptom": []}
def _sys_death():  return {"direct": ["死亡"], "symptom": []}
def _sys_dehyd():  return {"direct": ["脱水"], "symptom": []}
def _sys_sepsis(): return {"direct": ["敗血症"], "symptom": []}
def _sys_shock():  return {"direct": ["ショック", "アナフィラキシー"], "symptom": []}


def _match_adr(name: str):
    n = name.upper()
    if any(k in n for k in ("HEPAT", "LIVER", "JAUNDICE", "CHOLESTASIS", "CHOLANGITIS",
                              "AMINOTRANSFERASE INCREASED", "ASCITES")):
        return "HEP", _hep()
    if any(k in n for k in ("INTERSTITIAL LUNG", "INTERSTITIAL PNEUMONITIS",
                              "PULMONARY FIBROSIS", "LUNG DISORDER", "LUNG INFILTR",
                              "PNEUMONIA", "PULMONARY TOXICITY",
                              "PULMONARY ALVEOLAR HAEMORRHAGE")):
        return "ILD", _ild()
    if any(k in n for k in ("HYPOKAL", "PSEUDOALDOSTERONISM", "RHABDOMYOLYSIS", "MYOPATHY")):
        return "HPK", _hpk()
    if any(k in n for k in ("RASH", "ERYTHEMA", "URTICARIA", "ERUPTION", "TOXIC SKIN",
                              "STEVENS", "EPIDERMAL", "DERMATITIS", "PRURITUS",
                              "SKIN REACTION", "ECZEMA", "PUSTUL")):
        return "SKN", _skn()
    if any(k in n for k in ("APPETITE", "NAUSEA", "VOMIT", "DIARRHOEA", "DIARRHEA",
                              "ABDOMINAL PAIN", "CONSTIPAT", "ILEUS")):
        return "DIG", _dig()
    if any(k in n for k in ("HEADACHE", "DIZZINESS", "INSOMNIA", "SOMNOLENCE", "FATIGUE",
                              "MALAISE", "PARAESTHESIA", "PARESTHESIA",
                              "NEUROPATHY PERIPHERAL", "DELIRIUM", "LOSS OF CONSCIOUSNESS",
                              "PARALYSIS", "GAIT DISTURBANCE", "ABNORMAL BEHAV", "SCHIZOPH")):
        return "NS", _ns()
    if any(k in n for k in ("PALPITATIONS", "TACHYCARDIA", "ARRHYTHMIA", "HYPERTENS",
                              "HYPOTENS", "OEDEMA", "EDEMA", "MYOCARDIAL INFARCTION",
                              "HEART FAILURE")):
        return "CARD", _card()
    if any(k in n for k in ("HYPONATRAEMIA", "HYPONATREMIA", "HYPOCALCAEMIA", "HYPOCALCEMIA",
                              "HYPERCALCAEMIA", "HYPERCALCEMIA", "ELECTROLYTE")):
        return "ELE", _ele()
    if any(k in n for k in ("PLATELET", "THROMBOCYTOPENIA", "ANAEMIA", "ANEMIA",
                              "LEUKOCYTO", "LEUCOCYTO", "NEUTROPENIA", "PANCYTOPENIA")):
        return "BLD", _bld()
    if any(k in n for k in ("URINARY RETENTION", "RENAL FAILURE", "NEPHRITIS", "KIDNEY")):
        return "URO", _uro()
    if "CHILLS" in n: return "SYS", _sys_chills()
    if "DEATH" in n:  return "SYS", _sys_death()
    if "DEHYDRATION" in n: return "SYS", _sys_dehyd()
    if "SEPSIS" in n: return "SYS", _sys_sepsis()
    if any(k in n for k in ("SHOCK", "ANAPHYLACTIC", "ANAPHYLAXIS", "FEVER")):
        return "SYS", _sys_shock()
    return None, None


def _pdf_text(pdf_path: _pathlib.Path) -> str:
    out = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, check=True,
    )
    return out.stdout.decode("utf-8", errors="ignore")


def _side_effect_section(text: str) -> str:
    m = re.search(r"重大な副作用", text)
    if not m:
        m = re.search(r"副\s*作\s*用", text)
    if m:
        start = m.start()
        tail = text[start:]
        end_m = re.search(
            r"(?:用法及び用量|薬効薬理|保管上の注意|取扱い上|包装|包 装|承認条件|貯法|有効期間)",
            tail[20:],
        )
        end = 20 + end_m.start() if end_m else len(tail)
        return tail[:end]
    return text


def _scan(text: str, syn) -> bool:
    if syn is None:
        return False
    return any(text.count(p) > 0 for p in syn["direct"])


# ============================================================
# Main
# ============================================================

def main() -> int:
    with open(DATASET_PKL, "rb") as fh:
        ds = pickle.load(fh)
    df = ds["df"].reset_index(drop=True).copy()
    df["sample_idx"] = df.index.astype(int)
    df["label"] = df["label"].astype(int)

    with open(OOF_PKL, "rb") as fh:
        oof = pickle.load(fh)

    romaji = pd.read_csv(ROMAJI_CSV)[["TCMF_id", "formula_name_jp"]]
    adr_nodes = pd.read_csv(ADR_NODES)
    adr_name_col = [c for c in adr_nodes.columns if "name" in c.lower()][0]
    adr_name = dict(zip(adr_nodes["Adr_id"], adr_nodes[adr_name_col]))
    kanji_map = dict(zip(romaji["TCMF_id"], romaji["formula_name_jp"]))

    merged = df.merge(oof, on="sample_idx", how="inner")
    non_signal = merged[merged["label"] == 0].sort_values("prob", ascending=False).reset_index(drop=True)
    print(f"Universe: {len(non_signal)} non-signal pairs (label=0)")
    print(f"Score range: {non_signal['prob'].min():.4f} .. {non_signal['prob'].max():.4f}")
    print()

    # Cache PDF section per kanji.
    pdf_cache: dict[str, str | None] = {}
    def _section(kanji: str) -> str | None:
        if kanji in pdf_cache:
            return pdf_cache[kanji]
        path = INSERT_DIR / f"{kanji}.pdf"
        if not path.exists():
            pdf_cache[kanji] = None
            return None
        try:
            text = _pdf_text(path)
        except subprocess.CalledProcessError:
            pdf_cache[kanji] = None
            return None
        pdf_cache[kanji] = _side_effect_section(text)
        return pdf_cache[kanji]

    # Pre-compute Yes/No for every non-signal pair, only once.
    yes_flag = np.zeros(len(non_signal), dtype=bool)
    fam_arr = []
    print("Pre-computing PMDA flag for every non-signal pair...")
    for i, row in non_signal.iterrows():
        adr_pt = adr_name.get(row["Adr_id"], row["Adr_id"])
        fam, syn = _match_adr(str(adr_pt))
        fam_arr.append(fam or "UNMAPPED")
        if syn is None:
            continue
        kanji = kanji_map.get(row["TCMF_id"], "")
        if not kanji:
            continue
        sect = _section(kanji)
        if sect is None:
            continue
        yes_flag[i] = _scan(sect, syn)
    non_signal["pmda_yes"] = yes_flag
    non_signal["adr_family"] = fam_arr
    print(f"Pre-computation done; baseline rate over universe = "
          f"{yes_flag.mean()*100:.1f}%")
    print()

    # K grid.
    K_GRID = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, len(non_signal)]
    rng = np.random.default_rng(2024)

    print(f"{'K':>5} {'Top-K':>8} {'unstrat':>8} {'ADR-match':>10} "
          f"{'enr-unstrat':>12} {'enr-adrmatch':>13} {'pscore-min':>11}")
    print("-" * 78)
    rows = []
    for K in K_GRID:
        K = min(K, len(non_signal))
        top = non_signal.iloc[:K]
        top_yes = int(top["pmda_yes"].sum())
        top_rate = top_yes / K

        unstrat_idx = rng.choice(len(non_signal), size=K, replace=False)
        unstrat = non_signal.iloc[unstrat_idx]
        unstrat_yes = int(unstrat["pmda_yes"].sum())
        unstrat_rate = unstrat_yes / K

        # ADR-matched: for each ADR in the top-K, draw the same number of
        # rows uniformly from the non-signal universe restricted to that
        # ADR. If a given ADR has < n needed, draw what's available.
        wanted = Counter(top["Adr_id"].tolist())
        matched_idx = []
        for adr_id, n_need in wanted.items():
            avail = non_signal.index[non_signal["Adr_id"] == adr_id].to_numpy()
            if len(avail) <= n_need:
                matched_idx.extend(avail.tolist())
            else:
                matched_idx.extend(rng.choice(avail, size=n_need, replace=False).tolist())
        matched_idx = np.array(matched_idx, dtype=int)
        matched = non_signal.iloc[matched_idx]
        matched_yes = int(matched["pmda_yes"].sum())
        matched_rate = matched_yes / max(len(matched), 1)

        enr_unstrat = top_rate / unstrat_rate if unstrat_rate > 0 else float("inf")
        enr_match = top_rate / matched_rate if matched_rate > 0 else float("inf")

        score_min = top["prob"].min()

        print(f"{K:>5d} {top_rate*100:>7.1f}% {unstrat_rate*100:>7.1f}% "
              f"{matched_rate*100:>9.1f}% {enr_unstrat:>12.2f}x "
              f"{enr_match:>12.2f}x {score_min:>11.3f}")
        rows.append({
            "K": K,
            "top_yes": top_yes,
            "top_rate": top_rate,
            "unstrat_yes": unstrat_yes,
            "unstrat_rate": unstrat_rate,
            "matched_yes": matched_yes,
            "matched_n": len(matched),
            "matched_rate": matched_rate,
            "enr_unstrat": enr_unstrat,
            "enr_matched": enr_match,
            "min_score_in_topk": float(score_min),
        })

    out_df = pd.DataFrame(rows)
    out_csv = OOF_PKL.parent / "topk_pmda_enrichment_curve.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ADR family breakdown for top-K vs universe at K=200.
    print("\n=== ADR family composition by top-K ===")
    for K in [200, 500, 1000]:
        K = min(K, len(non_signal))
        top = non_signal.iloc[:K]
        print(f"\nTop-{K}:")
        for fam, sub in top.groupby("adr_family"):
            y = int(sub["pmda_yes"].sum())
            n = len(sub)
            pct_universe = (non_signal["adr_family"] == fam).sum() / len(non_signal)
            print(f"  {fam:<10} n={n:4d} ({100*n/K:5.1f}% of top-K, universe {100*pct_universe:4.1f}%)  "
                  f"Yes={y:3d}  rate={100*y/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
