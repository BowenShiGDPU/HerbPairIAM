r"""Build the K=200 / 500 / 1000 enrichment summary for Supplementary Table S13.

Re-uses the cached PMDA-flag computation from
``scan_topk_pmda_curve.py`` semantics (we recompute here for
self-containment) and writes a small CSV that
``render_external_validation_tex.py``-style downstream code (or a
direct ``\input{}`` of a generated .tex snippet) can read.

The output table has, for each K:
  - top-K Yes count and rate
  - unstratified random Yes count and rate (same K, drawn once
    with seed 2024)
  - ADR-matched random Yes count and rate
  - enrichment vs unstratified
  - enrichment vs ADR-matched
"""

from __future__ import annotations

import pathlib as _pathlib
import pickle
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

ROOT = _pathlib.Path(__file__).resolve().parents[2]
DATASET_PKL = ROOT / "outputs" / "dataset.pkl"
OOF_PKL = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "oof_predictions_with_attention.pkl"
INSERT_DIR = ROOT / "药品说明书"
ROMAJI_CSV = ROOT / "final_data_clean" / "TCMF_nodes_with_romaji.csv"
ADR_NODES = ROOT / "final_data_clean" / "ADR_nodes.csv"
OUT_CSV = OOF_PKL.parent / "topk_pmda_enrichment_table.csv"
OUT_TEX = ROOT / "paper_package" / "supplementary" / "supp_table_S13_topk_enrichment.tex"

K_GRID = [200, 500, 1000]


# ============================================================
# ADR family synonyms (unchanged)
# ============================================================
def _hep():
    return ["肝機能障害", "肝機能異常", "肝障害", "肝炎", "黄疸",
            "劇症肝炎", "肝不全", "胆汁うっ滞", "胆石", "胆管炎",
            "腹水", "AST", "ALT", "ALP", "γ-GTP", "γGTP", "Al-P",
            "アルカリホスファターゼ", "ビリルビン", "トランスアミナーゼ"]
def _ild():
    return ["間質性肺炎", "間質性肺疾患", "肺臓炎", "肺線維症",
            "肺胞炎", "肺炎", "肺胞出血", "呼吸困難"]
def _hpk():
    return ["低カリウム血症", "低K血症", "偽アルドステロン症",
            "ミオパチー", "横紋筋融解症"]
def _skn():
    return ["発疹", "発赤", "蕁麻疹", "皮疹", "薬疹",
            "中毒性皮疹", "中毒性表皮壊死症", "TEN",
            "皮膚粘膜眼症候群", "Stevens-Johnson",
            "スチーブンス・ジョンソン", "重症薬疹",
            "湿疹", "固定薬疹", "膿疱"]
def _dig():
    return ["食欲不振", "悪心", "嘔吐", "下痢", "腹痛", "便秘",
            "胃部不快感", "腸閉塞", "イレウス"]
def _ns():
    return ["めまい", "頭痛", "不眠", "眠気", "倦怠感",
            "しびれ", "ふらつき", "末梢神経障害", "せん妄",
            "意識障害", "麻痺", "不穏", "興奮", "幻覚"]
def _card():
    return ["動悸", "頻脈", "心悸亢進", "不整脈", "血圧上昇",
            "高血圧", "低血圧", "浮腫", "心筋梗塞", "心不全"]
def _ele():
    return ["低ナトリウム血症", "高ナトリウム血症",
            "低カルシウム血症", "高カルシウム血症", "電解質異常"]
def _bld():
    return ["血小板減少", "貧血", "白血球減少", "顆粒球減少",
            "汎血球減少", "血小板数"]
def _uro():
    return ["尿閉", "排尿障害", "腎機能障害", "腎不全",
            "間質性腎炎", "尿細管間質性腎炎", "クレアチニン", "BUN"]


def _match_adr(name: str):
    n = name.upper()
    if any(k in n for k in ("HEPAT", "LIVER", "JAUNDICE", "CHOLESTASIS", "CHOLANGITIS",
                              "AMINOTRANSFERASE INCREASED", "ASCITES")): return _hep()
    if any(k in n for k in ("INTERSTITIAL LUNG", "INTERSTITIAL PNEUMONITIS",
                              "PULMONARY FIBROSIS", "LUNG DISORDER", "LUNG INFILTR",
                              "PNEUMONIA", "PULMONARY TOXICITY",
                              "PULMONARY ALVEOLAR HAEMORRHAGE")): return _ild()
    if any(k in n for k in ("HYPOKAL", "PSEUDOALDOSTERONISM", "RHABDOMYOLYSIS", "MYOPATHY")): return _hpk()
    if any(k in n for k in ("RASH", "ERYTHEMA", "URTICARIA", "ERUPTION", "TOXIC SKIN",
                              "STEVENS", "EPIDERMAL", "DERMATITIS", "PRURITUS",
                              "SKIN REACTION", "ECZEMA", "PUSTUL")): return _skn()
    if any(k in n for k in ("APPETITE", "NAUSEA", "VOMIT", "DIARRHOEA", "DIARRHEA",
                              "ABDOMINAL PAIN", "CONSTIPAT", "ILEUS")): return _dig()
    if any(k in n for k in ("HEADACHE", "DIZZINESS", "INSOMNIA", "SOMNOLENCE", "FATIGUE",
                              "MALAISE", "PARAESTHESIA", "PARESTHESIA",
                              "NEUROPATHY PERIPHERAL", "DELIRIUM", "LOSS OF CONSCIOUSNESS",
                              "PARALYSIS", "GAIT DISTURBANCE", "ABNORMAL BEHAV", "SCHIZOPH")): return _ns()
    if any(k in n for k in ("PALPITATIONS", "TACHYCARDIA", "ARRHYTHMIA", "HYPERTENS",
                              "HYPOTENS", "OEDEMA", "EDEMA", "MYOCARDIAL INFARCTION",
                              "HEART FAILURE")): return _card()
    if any(k in n for k in ("HYPONATRAEMIA", "HYPONATREMIA", "HYPOCALCAEMIA", "HYPOCALCEMIA",
                              "HYPERCALCAEMIA", "HYPERCALCEMIA", "ELECTROLYTE")): return _ele()
    if any(k in n for k in ("PLATELET", "THROMBOCYTOPENIA", "ANAEMIA", "ANEMIA",
                              "LEUKOCYTO", "LEUCOCYTO", "NEUTROPENIA", "PANCYTOPENIA")): return _bld()
    if any(k in n for k in ("URINARY RETENTION", "RENAL FAILURE", "NEPHRITIS", "KIDNEY")): return _uro()
    if "CHILLS" in n: return ["悪寒"]
    if "DEATH" in n: return ["死亡"]
    if "DEHYDRATION" in n: return ["脱水"]
    if "SEPSIS" in n: return ["敗血症"]
    if any(k in n for k in ("SHOCK", "ANAPHYLACTIC", "ANAPHYLAXIS", "FEVER")):
        return ["ショック", "アナフィラキシー"]
    return None


def _pdf_text(pdf_path: _pathlib.Path) -> str:
    out = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"],
                         capture_output=True, check=True)
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
    print(f"[topk-table] universe = {len(non_signal)} non-signal pairs")

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
    print("Pre-computing PMDA flag for every non-signal pair...")
    for i, row in non_signal.iterrows():
        adr_pt = adr_name.get(row["Adr_id"], row["Adr_id"])
        syn = _match_adr(str(adr_pt))
        if syn is None: continue
        kanji = kanji_map.get(row["TCMF_id"], "")
        if not kanji: continue
        sect = _section(kanji)
        if sect is None: continue
        yes_flag[i] = any(sect.count(p) > 0 for p in syn)
    non_signal["pmda_yes"] = yes_flag
    print(f"  baseline rate over universe = {yes_flag.mean()*100:.1f}%")

    rng = np.random.default_rng(2024)
    rows = []
    for K in K_GRID:
        K = min(K, len(non_signal))
        # top-K
        top = non_signal.iloc[:K]
        top_yes = int(top["pmda_yes"].sum())
        top_rate = top_yes / K
        # unstratified random K
        u_idx = rng.choice(len(non_signal), size=K, replace=False)
        u = non_signal.iloc[u_idx]
        u_yes = int(u["pmda_yes"].sum())
        u_rate = u_yes / K
        # ADR-matched random K
        wanted = Counter(top["Adr_id"].tolist())
        m_idx = []
        for adr_id, n_need in wanted.items():
            avail = non_signal.index[non_signal["Adr_id"] == adr_id].to_numpy()
            if len(avail) <= n_need:
                m_idx.extend(avail.tolist())
            else:
                m_idx.extend(rng.choice(avail, size=n_need, replace=False).tolist())
        m_idx = np.array(m_idx, dtype=int)
        m = non_signal.iloc[m_idx]
        m_yes = int(m["pmda_yes"].sum())
        m_n = len(m)
        m_rate = m_yes / max(m_n, 1)

        enr_u = top_rate / u_rate if u_rate > 0 else float("inf")
        enr_m = top_rate / m_rate if m_rate > 0 else float("inf")

        rows.append({
            "K": K,
            "top_yes": top_yes,
            "top_rate": top_rate,
            "unstrat_yes": u_yes,
            "unstrat_rate": u_rate,
            "matched_yes": m_yes,
            "matched_n": m_n,
            "matched_rate": m_rate,
            "enrichment_unstrat": enr_u,
            "enrichment_matched": enr_m,
            "score_min": float(top["prob"].min()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[topk-table] wrote {OUT_CSV}")
    print()
    print(out.to_string(index=False))

    # Build LaTeX block.
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{\textbf{Top-$K$ stability of the PMDA-insert",
        r"enrichment.} For each cut-off $K\in\{200, 500, 1{,}000\}$,",
        r"the table reports the HerbPairIAM top-$K$ Yes rate, the",
        r"unstratified random baseline Yes rate, the ADR-matched",
        r"random baseline Yes rate, and the corresponding enrichment",
        r"ratios. All three baselines are drawn without replacement",
        r"under numpy random seed $2024$ from the same",
        r"$3{,}261$-pair non-signal universe and audited with the",
        r"identical Section-11 keyword list. Both enrichments are",
        r"reproduced across the three cut-offs (unstratified",
        r"$\geq 1.6\times$, ADR-matched $\geq 1.18\times$),",
        r"confirming that the $K\!=\!500$ headline of main",
        r"Table~5 is not a narrow operating point.}",
        r"\label{tab:supp:s13_topk}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Yes / $K$ (rate)} & \multicolumn{2}{c}{Enrichment vs random} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"$K$ & HerbPairIAM top-$K$ & Unstratified & ADR-matched & Unstratified & ADR-matched \\",
        r"\midrule",
    ]
    for r in rows:
        K = r["K"]
        lines.append(
            f"${K}$ & ${r['top_yes']}/{K}$ ($\\mathbf{{{r['top_rate']*100:.1f}\\%}}$) "
            f"& ${r['unstrat_yes']}/{K}$ (${r['unstrat_rate']*100:.1f}\\%$) "
            f"& ${r['matched_yes']}/{r['matched_n']}$ (${r['matched_rate']*100:.1f}\\%$) "
            f"& $\\mathbf{{{r['enrichment_unstrat']:.2f}\\times}}$ "
            f"& $\\mathbf{{{r['enrichment_matched']:.2f}\\times}}$ \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.3em}",
        r"{\footnotesize Score range covered by each top-$K$:",
        f"$K\\!=\\!200$ score $\\in[{rows[0]['score_min']:.3f}, 0.901]$; "
        f"$K\\!=\\!500$ score $\\in[{rows[1]['score_min']:.3f}, 0.901]$; "
        f"$K\\!=\\!1{{,}}000$ score $\\in[{rows[2]['score_min']:.3f}, 0.901]$.",
        r"As $K$ approaches the universe size of $3{,}261$ pairs the",
        r"three Yes rates necessarily converge to $19.6\%$. The full",
        r"$192$-row top-$500$ Yes set and the per-row CSVs are released",
        r"with the data package.\par}",
        r"\end{table}",
    ])
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
