"""Verify PMDA package-insert support for Supp Table~S9 (random sample).

Reads the random 15-row sample produced by
``build_random_external_validation_pool.py`` and scans each formula's
Japanese PMDA package insert for MedDRA-equivalent Japanese terms
corresponding to the predicted ADR. Matches are restricted to the
adverse-reaction section (Section~11, *fukusayou*) so that
indications (``effect and indication'' section) are not counted as
adverse reactions.

Decision rubric:

* ``Yes``: at least one direct PMDA Japanese term for the predicted
  ADR family appears in Section~11 of the formula's insert.
* ``No``: neither a direct term nor a strong-symptom term for the
  predicted ADR family appears in Section~11.

Outputs
-------
- Prints per-row evidence summary and a final count to stdout.
- Writes ``random_external_validation_verified.csv`` alongside the
  input sample with columns ``pmda_support`` (Yes/No) and
  ``pmda_evidence`` (matching terms and counts).
"""

from __future__ import annotations

import pathlib as _pathlib
import re
import subprocess

import pandas as pd


ROOT = _pathlib.Path(__file__).resolve().parents[2]
INSERT_DIR = ROOT / "药品说明书"
SAMPLE_CSV = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "random_external_validation_sample.csv"
OUT_CSV = SAMPLE_CSV.with_name("random_external_validation_verified.csv")


# Map ADR names to families with Japanese synonyms. "Drug-induced
# liver injury" and "Hepatic Function Abnormal" both collapse onto
# HEP because their PMDA Japanese terminology overlaps.
ADR_FAMILY = {
    "HEPATIC FUNCTION ABNORMAL": "HEP",
    "LIVER DISORDER":            "HEP",
    "Drug-induced liver injury": "HEP",
    "INTERSTITIAL LUNG DISEASE": "ILD",
    "HYPOKALAEMIA":              "HPK",
    "TOXIC SKIN ERUPTION":       "SKN",
}

ADR_SYNONYMS = {
    "HEP": {
        "direct": [
            "肝機能障害", "肝機能異常", "肝障害", "肝炎", "黄疸",
            "劇症肝炎", "肝不全",
            "AST", "ALT", "ALP", "γ-GTP", "γGTP", "Al-P",
            "アルカリホスファターゼ", "ビリルビン", "トランスアミナーゼ",
        ],
        "symptom": ["褐色尿", "全身倦怠感"],
    },
    "ILD": {
        "direct": ["間質性肺炎", "間質性肺疾患", "肺臓炎", "肺線維症", "肺胞炎"],
        "symptom": ["咳嗽", "呼吸困難"],
    },
    "HPK": {
        "direct": ["低カリウム血症", "低K血症", "偽アルドステロン症",
                   "ミオパチー", "横紋筋融解症"],
        "symptom": ["脱力感", "四肢けいれん"],
    },
    "SKN": {
        # Any skin-eruption term counts as direct PMDA evidence for
        # a generic "toxic skin eruption" prediction; the severe
        # named syndromes (SJS/TEN/etc.) are included but not
        # required.
        "direct": [
            "発疹", "発赤", "蕁麻疹", "皮疹", "薬疹",
            "中毒性皮疹", "中毒性表皮壊死症", "TEN",
            "皮膚粘膜眼症候群", "Stevens-Johnson",
            "スチーブンス・ジョンソン", "重症薬疹",
        ],
        "symptom": ["瘙痒"],
    },
}


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


def _scan(text: str, family: str) -> dict:
    syn = ADR_SYNONYMS[family]
    direct_hits = [(p, text.count(p)) for p in syn["direct"] if text.count(p) > 0]
    sym_hits    = [(p, text.count(p)) for p in syn["symptom"] if text.count(p) > 0]
    return {"direct": direct_hits, "symptom": sym_hits}


def _decide(hits: dict) -> str:
    return "Yes" if hits["direct"] else "No"


def main() -> int:
    sample = pd.read_csv(SAMPLE_CSV)
    verdicts = []
    print(f"{'Formula':<22} {'Kanji':<14} {'ADR':<30} {'Decision':<8} {'Evidence'}")
    print("-" * 130)
    for _, row in sample.iterrows():
        kanji = row["formula_kanji"]
        romaji = row["formula_romaji"]
        adr = str(row["ADR_name"]).strip()
        fam_key = adr.upper()
        family = None
        for k, v in ADR_FAMILY.items():
            if k.upper() == fam_key or k.upper() == adr.strip().upper():
                family = v
                break
        if family is None:
            family = "HEP" if "liver" in adr.lower() or "hepatic" in adr.lower() else None
        if family is None:
            print(f"{romaji:<22} {kanji:<14} {adr:<30} {'UNKNOWN':<8} (no ADR family mapping)")
            verdicts.append({**row.to_dict(),
                             "pmda_support": "Pending",
                             "pmda_evidence": "no ADR family mapping"})
            continue
        pdf_path = INSERT_DIR / f"{kanji}.pdf"
        if not pdf_path.exists():
            print(f"{romaji:<22} {kanji:<14} {adr:<30} {'MISSING':<8} (PDF not found)")
            verdicts.append({**row.to_dict(),
                             "pmda_support": "Pending",
                             "pmda_evidence": "PDF not found"})
            continue
        try:
            full_text = _pdf_text(pdf_path)
        except subprocess.CalledProcessError as exc:
            print(f"{romaji:<22} {kanji:<14} {adr:<30} {'ERROR':<8} ({exc})")
            verdicts.append({**row.to_dict(),
                             "pmda_support": "Pending",
                             "pmda_evidence": "extract error"})
            continue
        text = _side_effect_section(full_text)
        hits = _scan(text, family)
        decision = _decide(hits)
        evid = ", ".join(f"{p}x{n}" for p, n in hits["direct"][:4])
        if not evid and hits["symptom"]:
            evid = "[symptom] " + ", ".join(f"{p}x{n}" for p, n in hits["symptom"][:3])
        print(f"{romaji:<22} {kanji:<14} {adr:<30} {decision:<8} {evid}")
        verdicts.append({**row.to_dict(),
                         "pmda_support": decision,
                         "pmda_evidence": evid})
    print("-" * 130)
    n_yes = sum(1 for v in verdicts if v["pmda_support"] == "Yes")
    n_no  = sum(1 for v in verdicts if v["pmda_support"] == "No")
    print(f"\nYes = {n_yes}  No = {n_no}  Other = {len(verdicts) - n_yes - n_no}")
    pd.DataFrame(verdicts).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
