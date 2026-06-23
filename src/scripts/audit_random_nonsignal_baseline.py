"""PMDA baseline: audit 200 random non-signal formula-ADR pairs.

Parallel protocol to ``audit_top200_nonsignal.py`` but samples at
random from the non-signal (label=0) population. We draw TWO random
baselines so the enrichment interpretation is fair:

1. ``unstratified_200`` : 200 random pairs from the full non-signal
   universe (size ~3.3k). This reproduces the random-sample baseline
   suggested by the reviewer.
2. ``adr_matched_200`` : 200 random pairs whose ADR marginal
   distribution matches the top-200 non-signal set. This is the
   fair comparison: both samples see the same mix of ADR families
   so the hit-rate difference reflects HerbPairIAM's ranking, not
   a shift in the ADR pool.

ADR family mapping, Japanese keyword list, and Section-11
extraction regex are kept bit-identical to
``audit_top200_nonsignal.py``.
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
TOP200_CSV = OOF_PKL.parent / "top200_nonsignal_pmda_audit.csv"
INSERT_DIR = ROOT / "药品说明书"
ROMAJI_CSV = ROOT / "final_data_clean" / "TCMF_nodes_with_romaji.csv"
HERB_NODES = ROOT / "final_data_clean" / "CMM_nodes.csv"
ADR_NODES = ROOT / "final_data_clean" / "ADR_nodes.csv"
FORMULA_HERB = ROOT / "final_data_clean" / "CMM_TCMF.csv"

OUT_UNSTRAT_CSV = OOF_PKL.parent / "baseline_random_200_nonsignal_pmda_audit.csv"
OUT_MATCHED_CSV = OOF_PKL.parent / "baseline_adr_matched_200_nonsignal_pmda_audit.csv"

N_SAMPLE = 200
SEED = 2024


# ---- ADR family synonyms (IDENTICAL to audit_top200_nonsignal.py) ----

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


def _sys_chills():   return {"direct": ["悪寒"], "symptom": []}
def _sys_death():    return {"direct": ["死亡"], "symptom": []}
def _sys_dehyd():    return {"direct": ["脱水"], "symptom": []}
def _sys_sepsis():   return {"direct": ["敗血症"], "symptom": []}
def _sys_shock():    return {"direct": ["ショック", "アナフィラキシー"], "symptom": []}


def _match_adr(name: str) -> tuple[str, dict] | tuple[None, None]:
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
    if "CHILLS" in n:      return "SYS", _sys_chills()
    if "DEATH" in n:        return "SYS", _sys_death()
    if "DEHYDRATION" in n:  return "SYS", _sys_dehyd()
    if "SEPSIS" in n:       return "SYS", _sys_sepsis()
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
    stripped = re.sub(
        r"(?s)(?:効能\s*又は\s*効果|効能・効果).*?(?=(?:\d\s*\.\s*用法|用法・用量|用法及び用量|\n\d\s*\.))",
        "", text,
    )
    return stripped


def _scan(text: str, syn: dict) -> tuple[bool, str]:
    direct = [(p, text.count(p)) for p in syn["direct"] if text.count(p) > 0]
    if direct:
        return True, ", ".join(f"{p}x{n}" for p, n in direct[:5])
    sym = [(p, text.count(p)) for p in syn["symptom"] if text.count(p) > 0]
    if sym:
        return False, "[symptom] " + ", ".join(f"{p}x{n}" for p, n in sym[:3])
    return False, ""


def _row_to_attn(v, expected_len: int) -> np.ndarray:
    if v is None:
        return np.zeros(0, dtype=float)
    arr = np.asarray(v, dtype=float).ravel()
    if expected_len > 0 and arr.size != expected_len:
        if arr.size > expected_len:
            arr = arr[:expected_len]
        else:
            arr = np.pad(arr, (0, expected_len - arr.size))
    return arr


def _audit_sample(sample: pd.DataFrame, f2h: dict, herb_name: dict,
                    adr_name: dict, romaji_map: dict, kanji_map: dict,
                    pdf_cache: dict) -> pd.DataFrame:
    rows = []
    for _, entry in sample.iterrows():
        f_id = entry["TCMF_id"]
        a_id = entry["Adr_id"]
        adr_pt = adr_name.get(a_id, a_id)
        fam_key, syn = _match_adr(str(adr_pt))
        kanji = kanji_map.get(f_id, "")
        romaji_s = romaji_map.get(f_id, f_id)
        herbs = f2h.get(f_id, [])
        alpha = _row_to_attn(entry.get("herb_attn"), len(herbs))
        top_herb_idx = int(np.argmax(alpha)) if alpha.size > 0 else -1
        top_herb_id = herbs[top_herb_idx] if 0 <= top_herb_idx < len(herbs) else ""
        pair_attn = _row_to_attn(entry.get("pair_attn"), 0)
        expected_pairs = list(combinations(range(len(herbs)), 2))
        if pair_attn.size == len(expected_pairs) and pair_attn.size > 0:
            j = int(np.argmax(pair_attn))
            h1, h2 = expected_pairs[j]
            top_pair = f"{herb_name.get(herbs[h1], herbs[h1])} x {herb_name.get(herbs[h2], herbs[h2])}"
        else:
            top_pair = ""

        base = {
            "formula_romaji": romaji_s,
            "formula_kanji":  kanji,
            "predicted_adr":  adr_pt,
            "adr_family":     fam_key or "UNMAPPED",
            "score":          float(entry["prob"]),
            "top_herb":       herb_name.get(top_herb_id, top_herb_id),
            "top_pair":       top_pair,
        }
        if syn is None or not kanji:
            base.update({"pmda_support": "No", "pmda_evidence": ""})
        else:
            if kanji in pdf_cache:
                sect = pdf_cache[kanji]
            else:
                pdf_path = INSERT_DIR / f"{kanji}.pdf"
                if not pdf_path.exists():
                    sect = None
                else:
                    try:
                        sect = _side_effect_section(_pdf_text(pdf_path))
                    except subprocess.CalledProcessError:
                        sect = None
                pdf_cache[kanji] = sect
            if sect is None:
                base.update({"pmda_support": "No", "pmda_evidence": "[no insert]"})
            else:
                yes, evidence = _scan(sect, syn)
                base.update({"pmda_support": "Yes" if yes else "No",
                             "pmda_evidence": evidence})
        rows.append(base)
    out = pd.DataFrame(rows)
    out["score"] = out["score"].round(3)
    return out


def _summarise(out: pd.DataFrame, label: str) -> None:
    n_yes = int((out["pmda_support"] == "Yes").sum())
    print(f"\n[{label}] Yes = {n_yes} / {len(out)}  ({100 * n_yes / len(out):.1f}%)")
    print(f"[{label}] score range: "
          f"{out['score'].min():.3f} .. {out['score'].max():.3f}  (mean {out['score'].mean():.3f})")
    print(f"[{label}] Yes rate by ADR family:")
    for fam, sub in out.groupby("adr_family"):
        y = (sub["pmda_support"] == "Yes").sum()
        n = len(sub)
        print(f"   {fam:<9} Yes={y:3}/{n:3}  rate={100*y/n:5.1f}%")


def main() -> int:
    with open(DATASET_PKL, "rb") as fh:
        ds = pickle.load(fh)
    df = ds["df"].reset_index(drop=True).copy()
    df["sample_idx"] = df.index.astype(int)
    df["label"] = df["label"].astype(int)

    with open(OOF_PKL, "rb") as fh:
        oof = pickle.load(fh)

    romaji = pd.read_csv(ROMAJI_CSV)[["TCMF_id", "formula_name_jp", "formula_name_romaji"]]
    herb_nodes = pd.read_csv(HERB_NODES)
    adr_nodes = pd.read_csv(ADR_NODES)
    f_h = pd.read_csv(FORMULA_HERB)

    herb_name_col = ("herb_name_en" if "herb_name_en" in herb_nodes.columns
                     else [c for c in herb_nodes.columns if "_en" in c.lower()][0])
    herb_name = dict(zip(herb_nodes["CMM_id"], herb_nodes[herb_name_col]))
    adr_name_col = [c for c in adr_nodes.columns if "name" in c.lower()][0]
    adr_name = dict(zip(adr_nodes["Adr_id"], adr_nodes[adr_name_col]))
    f2h: dict[str, list[str]] = {}
    for fid, grp in f_h.groupby("TCMF_id"):
        f2h[fid] = sorted(grp["CMM_id"].tolist())
    romaji_map = dict(zip(romaji["TCMF_id"], romaji["formula_name_romaji"]))
    kanji_map = dict(zip(romaji["TCMF_id"], romaji["formula_name_jp"]))

    merged = df.merge(oof, on="sample_idx", how="inner")
    non_signal = merged[merged["label"] == 0].reset_index(drop=True)
    print(f"[baseline] non-signal universe size = {len(non_signal)}")

    rng = np.random.default_rng(SEED)

    # -----------------------------------------------------------
    # Baseline 1: unstratified random 200
    # -----------------------------------------------------------
    idx1 = rng.choice(len(non_signal), size=N_SAMPLE, replace=False)
    sample1 = non_signal.iloc[np.sort(idx1)].reset_index(drop=True)
    pdf_cache: dict[str, str | None] = {}
    out1 = _audit_sample(sample1, f2h, herb_name, adr_name, romaji_map, kanji_map, pdf_cache)
    out1.to_csv(OUT_UNSTRAT_CSV, index=False)
    _summarise(out1, "unstratified")
    print(f"[baseline] wrote {OUT_UNSTRAT_CSV}")

    # -----------------------------------------------------------
    # Baseline 2: ADR-matched to top-200 distribution
    # -----------------------------------------------------------
    top200 = pd.read_csv(TOP200_CSV)
    top_adr_counts = Counter(top200["predicted_adr"].astype(str).tolist())
    print(f"\n[matched] top-200 has {len(top_adr_counts)} distinct ADRs")
    # For each ADR in top-200, draw the same count from the non-signal
    # pool restricted to that ADR (uniform over formulas).
    idx_pool = {}
    for adr_pt, n_need in top_adr_counts.items():
        avail = non_signal[
            non_signal["Adr_id"].map(lambda a: adr_name.get(a, a)) == adr_pt
        ]
        if len(avail) == 0:
            print(f"   [matched] ADR '{adr_pt}' has no non-signal pool rows. Skipping.")
            continue
        k = min(n_need, len(avail))
        draw = rng.choice(len(avail), size=k, replace=False)
        idx_pool[adr_pt] = avail.iloc[draw]
    sample2 = pd.concat(idx_pool.values(), ignore_index=True)
    print(f"[matched] drew {len(sample2)} random pairs ADR-matched to top-200")

    pdf_cache2: dict[str, str | None] = {}
    out2 = _audit_sample(sample2, f2h, herb_name, adr_name, romaji_map, kanji_map, pdf_cache2)
    out2.to_csv(OUT_MATCHED_CSV, index=False)
    _summarise(out2, "ADR-matched")
    print(f"[matched] wrote {OUT_MATCHED_CSV}")

    # -----------------------------------------------------------
    # Enrichment summary
    # -----------------------------------------------------------
    top_yes = int((top200["pmda_support"] == "Yes").sum())
    top_n = len(top200)
    random_yes = int((out1["pmda_support"] == "Yes").sum())
    matched_yes = int((out2["pmda_support"] == "Yes").sum())
    print()
    print("======================================================================")
    print("Enrichment summary")
    print("======================================================================")
    print(f"  Top-200 (HerbPairIAM rank): {top_yes}/{top_n}  = {100*top_yes/top_n:.1f}%")
    print(f"  Random 200 (unstratified):   {random_yes}/{N_SAMPLE}  = {100*random_yes/N_SAMPLE:.1f}%")
    print(f"  Random 200 (ADR-matched):    {matched_yes}/{len(out2)}  = {100*matched_yes/len(out2):.1f}%")
    if matched_yes > 0:
        print(f"  Enrichment (top / ADR-matched): {(top_yes/top_n) / (matched_yes/len(out2)):.2f}x")
    if random_yes > 0:
        print(f"  Enrichment (top / unstratified): {(top_yes/top_n) / (random_yes/N_SAMPLE):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
