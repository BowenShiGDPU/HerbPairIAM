"""Render Main Table 5 and Supp Table S9 from the top-200 PMDA audit CSV.

Inputs
------
results/.../interpretability/top200_nonsignal_pmda_audit.csv
    Output of ``audit_top200_nonsignal.py``; 8 columns incl.\\
    ``formula_romaji``, ``predicted_adr``, ``adr_family``, ``score``,
    ``top_herb``, ``top_pair``, ``pmda_support``, ``pmda_evidence``.

Outputs
-------
paper_package/main/tables/table5_external_validation_family.tex
    Family-level aggregate Yes/No breakdown (Main Table 5).
paper_package/supplementary/supp_table_S9_external_validation.tex
    Full list of every \\textsc{Yes} row (top-200 audit).
Both LaTeX blocks use pure ASCII column content so they compile without
CJK support.
"""

from __future__ import annotations

import pathlib as _pathlib
import pandas as pd


ROOT = _pathlib.Path(__file__).resolve().parents[2]
SRC_CSV = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "top200_nonsignal_pmda_audit.csv"
OUT_MAIN = ROOT / "paper_package" / "main" / "tables" / "table5_external_validation_family.tex"
OUT_SUPP = ROOT / "paper_package" / "supplementary" / "supp_table_S9_external_validation.tex"

# Family code -> human-readable label for Main Table 5. Order here is
# the display order (by descending Yes rate within size tiers).
FAMILY_LABELS = [
    ("SKN",  "Skin eruption"),
    ("NS",   "Nervous system"),
    ("HPK",  "Hypokalaemia (pseudoaldosteronism, myopathy)"),
    ("DIG",  "Digestive (appetite, nausea, diarrhoea, ileus)"),
    ("HEP",  "Hepatic (liver dysfunction, jaundice, cholestasis)"),
    ("ILD",  "Interstitial lung disease (pneumonia, fibrosis)"),
    ("OTHER","Other (blood, cardiac, electrolyte, systemic, urinary)"),
]
OTHER_SET = {"BLD", "CARD", "ELE", "SYS", "URO"}


# Transliterate matched Japanese terms in the ``pmda_evidence`` column
# so the Supp table compiles without CJK support. This mapping is
# bounded to the terms we emit in the scanner.
JP_TRANSLIT = {
    "肝機能障害":   "hepatic dysfunction",
    "肝機能異常":   "hepatic abnormality",
    "肝障害":       "liver injury",
    "肝炎":         "hepatitis",
    "黄疸":         "jaundice",
    "劇症肝炎":     "fulminant hepatitis",
    "肝不全":       "hepatic failure",
    "胆汁うっ滞":   "cholestasis",
    "胆石":         "cholelithiasis",
    "胆管炎":       "cholangitis",
    "腹水":         "ascites",
    "アルカリホスファターゼ": "ALP",
    "ビリルビン":   "bilirubin",
    "トランスアミナーゼ": "transaminase",
    "間質性肺炎":   "interstitial pneumonia",
    "間質性肺疾患": "interstitial lung disease",
    "肺臓炎":       "pneumonitis",
    "肺線維症":     "pulmonary fibrosis",
    "肺胞炎":       "alveolitis",
    "肺炎":         "pneumonia",
    "肺胞出血":     "alveolar haemorrhage",
    "呼吸困難":     "dyspnoea",
    "低カリウム血症": "hypokalaemia",
    "低K血症":      "hypokalaemia",
    "偽アルドステロン症": "pseudoaldosteronism",
    "ミオパチー":   "myopathy",
    "横紋筋融解症": "rhabdomyolysis",
    "発疹":         "rash",
    "発赤":         "erythema",
    "蕁麻疹":       "urticaria",
    "皮疹":         "skin eruption",
    "薬疹":         "drug eruption",
    "中毒性皮疹":   "toxic eruption",
    "中毒性表皮壊死症": "TEN",
    "皮膚粘膜眼症候群": "SJS",
    "重症薬疹":     "severe drug eruption",
    "湿疹":         "eczema",
    "固定薬疹":     "fixed drug eruption",
    "膿疱":         "pustule",
    "瘙痒":         "pruritus",
    "食欲不振":     "anorexia",
    "悪心":         "nausea",
    "嘔吐":         "vomiting",
    "下痢":         "diarrhoea",
    "腹痛":         "abdominal pain",
    "便秘":         "constipation",
    "胃部不快感":   "epigastric discomfort",
    "腸閉塞":       "ileus",
    "イレウス":     "ileus",
    "めまい":       "dizziness",
    "頭痛":         "headache",
    "不眠":         "insomnia",
    "眠気":         "somnolence",
    "倦怠感":       "malaise",
    "しびれ":       "paraesthesia",
    "ふらつき":     "unsteadiness",
    "末梢神経障害": "peripheral neuropathy",
    "せん妄":       "delirium",
    "意識障害":     "impaired consciousness",
    "麻痺":         "paralysis",
    "不穏":         "agitation",
    "興奮":         "agitation",
    "幻覚":         "hallucination",
    "動悸":         "palpitations",
    "頻脈":         "tachycardia",
    "心悸亢進":     "palpitations",
    "不整脈":       "arrhythmia",
    "血圧上昇":     "hypertension",
    "高血圧":       "hypertension",
    "低血圧":       "hypotension",
    "浮腫":         "oedema",
    "心筋梗塞":     "myocardial infarction",
    "心不全":       "heart failure",
    "AST":          "AST",
    "ALT":          "ALT",
    "ALP":          "ALP",
    "Al-P":         "ALP",
    "γ-GTP":        "gamma-GTP",
    "γGTP":         "gamma-GTP",
}


def _transliterate(evidence: str) -> str:
    """Replace Japanese tokens in a ``term x count`` evidence string."""
    if not isinstance(evidence, str) or not evidence.strip():
        return ""
    if evidence.startswith("[symptom only] "):
        body = evidence.removeprefix("[symptom only] ")
        return "[symptom] " + _transliterate(body)
    parts: list[str] = []
    for chunk in evidence.split(", "):
        # chunk form: 'term xN'
        if "x" in chunk:
            term, _, cnt = chunk.rpartition("x")
            term = term.strip()
            en = JP_TRANSLIT.get(term, term)
            parts.append(f"{en}$\\times${cnt}")
        else:
            parts.append(chunk)
    return ", ".join(parts)


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _build_family_table(df: pd.DataFrame) -> str:
    """Main Table 5: 8-row family-level aggregate."""
    # Remap per-row family into the display buckets (OTHER absorbs
    # BLD/CARD/ELE/SYS/URO).
    display_family = df["adr_family"].where(
        ~df["adr_family"].isin(OTHER_SET), "OTHER"
    )
    df2 = df.assign(display_family=display_family)

    lines = [
        r"\begin{table*}[t]",
        r"\caption{\textbf{External validation on 200 top-scoring",
        r"non-signal HerbPairIAM predictions.} For each",
        r"out-of-fold label $=\!0$ candidate with $\hat{p}\geq 0.5$, we",
        r"machine-extracted Section~11 (fukusayou) of the",
        r"formula-specific Japanese PMDA package insert and searched",
        r"for MedDRA-equivalent Japanese terms for the predicted ADR",
        r"family. A prediction is counted as PMDA-supported",
        r"(\textsc{Yes}) only when a direct Japanese term for the",
        r"predicted family appears in the insert's adverse-reaction",
        r"section. The aggregate hit rate of $33.5\%$ on 200 novel",
        r"candidates substantially exceeds the base rate of PMDA-insert",
        r"coverage of random formula--ADR pairs (near zero) and",
        r"concentrates in the ADR families that are broadly listed on",
        r"Kampo inserts (skin, hypokalaemia, nervous system), whereas",
        r"severe hepatic and interstitial lung reactions, which are",
        r"only listed on a small set of formulas with confirmed",
        r"regulatory signals, contribute the bulk of the remaining",
        r"\textsc{No} rows and therefore the pharmacovigilance",
        r"hypotheses. See Supp Table~S9 for all",
        r"$67$ \textsc{Yes} rows with verbatim PMDA evidence.}",
        r"\label{tab:external_validation_family}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"ADR family & $n$ & Yes & Yes rate \\",
        r"\midrule",
    ]
    total_n = total_y = 0
    for code, label in FAMILY_LABELS:
        if code == "OTHER":
            sub = df2[df2["display_family"] == "OTHER"]
        else:
            sub = df2[df2["display_family"] == code]
        n = len(sub)
        y = int((sub["pmda_support"] == "Yes").sum())
        total_n += n
        total_y += y
        rate = f"{100 * y / n:.1f}\\%" if n else "--"
        lines.append(f"{label} & {n} & {y} & {rate} \\\\")
    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total_n}}} & "
        f"\\textbf{{{total_y}}} & \\textbf{{{100 * total_y / total_n:.1f}\\%}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.3em}",
        r"{\footnotesize \textit{Methodology.} The pool of $200$ candidate",
        r"formula--ADR pairs is the complete set of out-of-fold HerbPairIAM",
        r"predictions on non-signal pairs (training label $=\!0$) with",
        r"predicted probability $\geq 0.5$ in the canonical seed-$42$",
        r"10-fold run. Scripted PMDA audit follows",
        r"\texttt{src/scripts/audit\_top200\_nonsignal.py}; the full",
        r"$200$-row CSV is released with the data package, and every",
        r"\textsc{Yes} row is reproduced with its matched Japanese",
        r"term and count in Supp Table~S9.\par}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _build_supp_all_yes(df: pd.DataFrame) -> str:
    """Supp Table S9: every Yes row with evidence (transliterated)."""
    yes = df[df["pmda_support"] == "Yes"].sort_values("score", ascending=False).reset_index(drop=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{\textbf{External validation: all $67$ PMDA-supported",
        r"rows from the top-$200$ non-signal audit} (Main",
        r"Table~\protect\ref*{tab:external_validation_family}). Each row",
        r"is a HerbPairIAM out-of-fold prediction on a formula--ADR pair",
        r"not used as a training positive, whose formula-specific",
        r"Japanese PMDA package insert lists the predicted ADR family",
        r"in its Section~11 adverse-reaction block. Japanese terms have",
        r"been transliterated to English MedDRA-equivalent wording for",
        r"readability; the verbatim Japanese tokens are kept in the",
        r"released CSV.}",
        r"\label{tab:supp:s9_external_yes_all}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llccl}",
        r"\toprule",
        r"Formula & Predicted ADR & Score & Top herb & PMDA-insert evidence \\",
        r"\midrule",
    ]
    for _, row in yes.iterrows():
        romaji = _latex_escape(row["formula_romaji"])
        adr = _latex_escape(row["predicted_adr"]).title()
        score = f"{float(row['score']):.3f}"
        top_herb = _latex_escape(row["top_herb"])
        evidence = _transliterate(str(row["pmda_evidence"]))
        lines.append(
            f"{romaji} & {adr} & ${score}$ & {top_herb} & {evidence} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{0.3em}",
        r"{\footnotesize Score = HerbPairIAM predicted probability,",
        r"pooled out-of-fold over the canonical 10-fold run",
        r"(seed $=\!42$). Top herb = attention-ranked top-1 herb.",
        r"The $133$ \textsc{No} rows of the top-$200$ audit are released",
        r"with the data package.\par}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    df = pd.read_csv(SRC_CSV)
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUPP.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAIN.write_text(_build_family_table(df))
    OUT_SUPP.write_text(_build_supp_all_yes(df))
    y = int((df["pmda_support"] == "Yes").sum())
    n = len(df)
    print(f"[render] wrote {OUT_MAIN}")
    print(f"[render] wrote {OUT_SUPP}  ({y} yes rows / {n} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
