"""Audit the top-200 non-signal HerbPairIAM predictions against PMDA inserts.

Produces a single CSV in the exact format of Supp Table~S9:

    formula_romaji, formula_kanji, predicted_adr, score,
    top_herb, top_pair, pmda_support, pmda_evidence

Rows are sorted by score descending. ``pmda_support`` is ``Yes`` iff
the formula-specific Japanese PMDA package insert lists a direct
Japanese term for the predicted ADR family inside Section~11
(fukusayou). ``pmda_evidence`` gives the actual matched terms.
"""

from __future__ import annotations

import pathlib as _pathlib
import pickle
import re
import subprocess
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
OUT_CSV = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "top200_nonsignal_pmda_audit.csv"

TOP_N = 200


# ============================================================
# ADR family synonyms -- deliberately permissive, covering
# MedDRA PT variants and PMDA Japanese insert wording.
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
    return {
        "direct": ["間質性肺炎", "間質性肺疾患", "肺臓炎", "肺線維症",
                   "肺胞炎", "肺炎", "肺胞出血", "呼吸困難"],
        "symptom": [],
    }

def _hpk():
    return {
        "direct": ["低カリウム血症", "低K血症", "偽アルドステロン症",
                   "ミオパチー", "横紋筋融解症"],
        "symptom": [],
    }

def _skn():
    return {
        "direct": ["発疹", "発赤", "蕁麻疹", "皮疹", "薬疹",
                   "中毒性皮疹", "中毒性表皮壊死症", "TEN",
                   "皮膚粘膜眼症候群", "Stevens-Johnson",
                   "スチーブンス・ジョンソン", "重症薬疹",
                   "湿疹", "固定薬疹", "膿疱"],
        "symptom": ["瘙痒"],
    }

def _dig():
    """Digestive: appetite, nausea, diarrhoea, vomiting, ileus."""
    return {
        "direct": ["食欲不振", "悪心", "嘔吐", "下痢", "腹痛", "便秘",
                   "胃部不快感", "腸閉塞", "イレウス"],
        "symptom": [],
    }

def _ns():
    """Generic nervous-system / CNS."""
    return {
        "direct": ["めまい", "頭痛", "不眠", "眠気", "倦怠感",
                   "しびれ", "ふらつき", "末梢神経障害", "せん妄",
                   "意識障害", "麻痺", "不穏", "興奮", "幻覚"],
        "symptom": [],
    }

def _card():
    return {
        "direct": ["動悸", "頻脈", "心悸亢進", "不整脈", "血圧上昇",
                   "高血圧", "低血圧", "浮腫", "心筋梗塞", "心不全"],
        "symptom": [],
    }

def _ele():
    """Electrolyte imbalance other than potassium."""
    return {
        "direct": ["低ナトリウム血症", "高ナトリウム血症",
                   "低カルシウム血症", "高カルシウム血症",
                   "電解質異常"],
        "symptom": [],
    }

def _bld():
    """Haematology."""
    return {
        "direct": ["血小板減少", "貧血", "白血球減少", "顆粒球減少",
                   "汎血球減少", "血小板数"],
        "symptom": [],
    }

def _uro():
    """Urinary / renal."""
    return {
        "direct": ["尿閉", "排尿障害", "腎機能障害", "腎不全",
                   "間質性腎炎", "尿細管間質性腎炎", "クレアチニン",
                   "BUN"],
        "symptom": [],
    }

def _sys_chills():
    return {"direct": ["悪寒"], "symptom": []}


def _sys_death():
    return {"direct": ["死亡"], "symptom": []}


def _sys_dehydration():
    return {"direct": ["脱水"], "symptom": []}


def _sys_sepsis():
    return {"direct": ["敗血症"], "symptom": []}


def _sys_shock():
    return {"direct": ["ショック", "アナフィラキシー"], "symptom": []}


def _match_adr(name: str) -> tuple[str, dict] | tuple[None, None]:
    n = name.upper()
    # Hepatic: broadened to substring 'HEPAT' and 'CHOL' to catch
    # HEPATOCELLULAR INJURY, Cholangitis, cholestasis, AST-INCREASED.
    if any(k in n for k in (
        "HEPAT", "LIVER", "JAUNDICE", "CHOLESTASIS", "CHOLANGITIS",
        "AMINOTRANSFERASE INCREASED", "ASCITES",
    )):
        return "HEP", _hep()
    if any(k in n for k in (
        "INTERSTITIAL LUNG", "INTERSTITIAL PNEUMONITIS",
        "PULMONARY FIBROSIS", "LUNG DISORDER", "LUNG INFILTR",
        "PNEUMONIA", "PULMONARY TOXICITY",
        "PULMONARY ALVEOLAR HAEMORRHAGE",
    )):
        return "ILD", _ild()
    if any(k in n for k in (
        "HYPOKAL", "PSEUDOALDOSTERONISM", "RHABDOMYOLYSIS", "MYOPATHY",
    )):
        return "HPK", _hpk()
    # Skin: broadened to just 'ERUPTION', 'ECZEMA', 'PUSTUL'.
    if any(k in n for k in (
        "RASH", "ERYTHEMA", "URTICARIA", "ERUPTION", "TOXIC SKIN",
        "STEVENS", "EPIDERMAL", "DERMATITIS", "PRURITUS",
        "SKIN REACTION", "ECZEMA", "PUSTUL",
    )):
        return "SKN", _skn()
    if any(k in n for k in (
        "APPETITE", "NAUSEA", "VOMIT", "DIARRHOEA", "DIARRHEA",
        "ABDOMINAL PAIN", "CONSTIPAT", "ILEUS",
    )):
        return "DIG", _dig()
    if any(k in n for k in (
        "HEADACHE", "DIZZINESS", "INSOMNIA", "SOMNOLENCE", "FATIGUE",
        "MALAISE", "PARAESTHESIA", "PARESTHESIA",
        "NEUROPATHY PERIPHERAL", "DELIRIUM", "LOSS OF CONSCIOUSNESS",
        "PARALYSIS", "GAIT DISTURBANCE", "ABNORMAL BEHAV", "SCHIZOPH",
    )):
        return "NS", _ns()
    if any(k in n for k in (
        "PALPITATIONS", "TACHYCARDIA", "ARRHYTHMIA", "HYPERTENS",
        "HYPOTENS", "OEDEMA", "EDEMA", "MYOCARDIAL INFARCTION",
        "HEART FAILURE",
    )):
        return "CARD", _card()
    if any(k in n for k in (
        "HYPONATRAEMIA", "HYPONATREMIA", "HYPOCALCAEMIA", "HYPOCALCEMIA",
        "HYPERCALCAEMIA", "HYPERCALCEMIA", "ELECTROLYTE",
    )):
        return "ELE", _ele()
    if any(k in n for k in (
        "PLATELET", "THROMBOCYTOPENIA", "ANAEMIA", "ANEMIA",
        "LEUKOCYTO", "LEUCOCYTO", "NEUTROPENIA", "PANCYTOPENIA",
    )):
        return "BLD", _bld()
    if any(k in n for k in (
        "URINARY RETENTION", "RENAL FAILURE", "NEPHRITIS", "KIDNEY",
    )):
        return "URO", _uro()
    if "CHILLS" in n:
        return "SYS", _sys_chills()
    if "DEATH" in n:
        return "SYS", _sys_death()
    if "DEHYDRATION" in n:
        return "SYS", _sys_dehydration()
    if "SEPSIS" in n:
        return "SYS", _sys_sepsis()
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
        return False, "[symptom only] " + ", ".join(f"{p}x{n}" for p, n in sym[:3])
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
                     else "CMM_name_EN" if "CMM_name_EN" in herb_nodes.columns
                     else [c for c in herb_nodes.columns if "_en" in c.lower()][0])
    herb_name = dict(zip(herb_nodes["CMM_id"], herb_nodes[herb_name_col]))
    adr_name_col = [c for c in adr_nodes.columns if "name" in c.lower()][0]
    adr_name = dict(zip(adr_nodes["Adr_id"], adr_nodes[adr_name_col]))
    f2h: dict[str, list[str]] = {}
    for fid, grp in f_h.groupby("TCMF_id"):
        f2h[fid] = sorted(grp["CMM_id"].tolist())

    merged = df.merge(oof, on="sample_idx", how="inner")
    non_signal = merged[merged["label"] == 0].sort_values("prob", ascending=False).head(TOP_N).reset_index(drop=True)
    print(f"[top200] built non-signal top-{TOP_N}; "
          f"prob range [{non_signal['prob'].min():.3f}, {non_signal['prob'].max():.3f}]",
          flush=True)

    romaji_map = dict(zip(romaji["TCMF_id"], romaji["formula_name_romaji"]))
    kanji_map = dict(zip(romaji["TCMF_id"], romaji["formula_name_jp"]))

    # Cache PDF reads so we don't extract the same insert repeatedly.
    pdf_cache: dict[str, str] = {}
    def _get_side_effect(kanji: str) -> str | None:
        if kanji in pdf_cache:
            return pdf_cache[kanji]
        pdf_path = INSERT_DIR / f"{kanji}.pdf"
        if not pdf_path.exists():
            pdf_cache[kanji] = None
            return None
        try:
            text = _pdf_text(pdf_path)
        except subprocess.CalledProcessError:
            pdf_cache[kanji] = None
            return None
        pdf_cache[kanji] = _side_effect_section(text)
        return pdf_cache[kanji]

    rows = []
    n_yes = n_no = n_missing = n_unmapped = 0
    for _, entry in non_signal.iterrows():
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
            top_pair_idx = int(np.argmax(pair_attn))
            h1_idx, h2_idx = expected_pairs[top_pair_idx]
            top_pair = (f"{herb_name.get(herbs[h1_idx], herbs[h1_idx])}"
                         f" x {herb_name.get(herbs[h2_idx], herbs[h2_idx])}")
        else:
            top_pair = ""

        base = {
            "formula_romaji":  romaji_s,
            "formula_kanji":   kanji,
            "predicted_adr":   adr_pt,
            "adr_family":      fam_key or "other",
            "score":           float(entry["prob"]),
            "top_herb":        herb_name.get(top_herb_id, top_herb_id),
            "top_pair":        top_pair,
        }

        if syn is None:
            base.update({"pmda_support": "Unmapped ADR family",
                         "pmda_evidence": ""})
            n_unmapped += 1
        elif not kanji:
            base.update({"pmda_support": "No Kanji mapping",
                         "pmda_evidence": ""})
            n_missing += 1
        else:
            sect = _get_side_effect(kanji)
            if sect is None:
                base.update({"pmda_support": "PDF missing",
                             "pmda_evidence": ""})
                n_missing += 1
            else:
                yes, evidence = _scan(sect, syn)
                base.update({"pmda_support": "Yes" if yes else "No",
                             "pmda_evidence": evidence})
                if yes:
                    n_yes += 1
                else:
                    n_no += 1
        rows.append(base)

    out = pd.DataFrame(rows)
    out["score"] = out["score"].round(3)
    out.to_csv(OUT_CSV, index=False)

    print(f"\n[top200] wrote {OUT_CSV}  ({len(out)} rows)")
    print(f"[top200] Yes = {n_yes}   No = {n_no}   "
          f"Unmapped = {n_unmapped}   Missing = {n_missing}")
    print()
    print("ADR family breakdown (Yes / Total by family):")
    for fam, sub in out.groupby("adr_family"):
        y = (sub["pmda_support"] == "Yes").sum()
        n = len(sub)
        print(f"  {fam:<8} {y}/{n}")
    print()
    print("Top 20 Yes rows:")
    yes_rows = out[out["pmda_support"] == "Yes"].sort_values("score", ascending=False).head(20)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(yes_rows[[
            "formula_romaji", "predicted_adr", "score", "top_herb",
            "pmda_evidence",
        ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
