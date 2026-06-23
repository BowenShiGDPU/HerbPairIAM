r"""Check whether the 6 old-paper ``Yes'' formula--ADR pairs are
predicted positive by the current HerbPairIAM model.

Looks up each old-paper pair (formula romaji + ADR PT) in the
canonical 10-fold OOF predictions and reports:
* current HerbPairIAM OOF probability,
* whether the pair was in the training signal corpus (label),
* whether the pair was at least in the tested-universe
  evaluation set (tested_source),
* predicted rank at the operating threshold we use elsewhere
  in the paper (prob >= 0.5, and prob >= prevalence 0.178).

If a pair is not in the evaluation set at all (e.g.\ because
the formula is not covered by the pair-level features), we
print ``(not evaluated)''.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_PKL = ROOT / "outputs" / "dataset.pkl"
OOF_PKL = ROOT / "results" / "formal_doseaware_neg10_auroc" / "main_benchmark" / "interpretability" / "oof_predictions_with_attention.pkl"


OLD_PAPER_YES = [
    # (romaji,           TCMF_id,  ADR PT,                     Adr_id,    old_score)
    ("Shoseiryuto",      "TCMF76", "DECREASED APPETITE",       "Adr4805", 0.940),
    ("Junchoto",         "TCMF72", "HEPATIC FUNCTION ABNORMAL","Adr1414", 0.904),
    ("Goreisan",         "TCMF48", "RASH",                      "Adr2621", 0.838),
    ("Daikenchuto",      "TCMF93", "DIARRHOEA",                 "Adr908",  0.784),
    ("Ryokeijutsukanto", "TCMF147","HYPOKALAEMIA",              "Adr1557", 0.730),
    ("Kakkonto",         "TCMF14", "NAUSEA",                    "Adr2095", 0.635),
]


def main() -> int:
    with open(DATASET_PKL, "rb") as fh:
        ds = pickle.load(fh)
    df = ds["df"].reset_index(drop=True).copy()
    df["sample_idx"] = df.index.astype(int)
    df["label"] = df["label"].astype(int)

    with open(OOF_PKL, "rb") as fh:
        oof = pickle.load(fh)

    merged = df.merge(oof, on="sample_idx", how="left")

    print(f"{'Formula':<20} {'ADR':<28} {'old score':>10} "
          f"{'new prob':>10} {'label':>6} {'source':<12} verdict")
    print("-" * 105)

    evaluated_and_high = 0
    evaluated_and_low = 0
    not_evaluated = 0

    for romaji, tcmf, adr_pt, adr_id, old_score in OLD_PAPER_YES:
        row = merged[(merged["TCMF_id"] == tcmf) & (merged["Adr_id"] == adr_id)]
        if row.empty:
            print(f"{romaji:<20} {adr_pt:<28} {old_score:>10.3f} "
                  f"{'(not in ds)':>10} {'-':>6} {'-':<12} not evaluated")
            not_evaluated += 1
            continue
        row = row.iloc[0]
        prob = row.get("prob")
        label = int(row["label"])
        source = str(row.get("tested_source", ""))
        if pd.isna(prob):
            verdict = "no OOF prediction"
            not_evaluated += 1
            prob_s = "(no OOF)"
        else:
            prob = float(prob)
            prob_s = f"{prob:.3f}"
            if prob >= 0.5:
                verdict = "PREDICTED positive (>=0.5)"
                evaluated_and_high += 1
            elif prob >= 0.178:
                verdict = "above prevalence (0.178) but <0.5"
                evaluated_and_high += 1
            else:
                verdict = "below threshold"
                evaluated_and_low += 1
        print(f"{romaji:<20} {adr_pt:<28} {old_score:>10.3f} "
              f"{prob_s:>10} {label:>6} {source:<12} {verdict}")
    print("-" * 105)
    print(f"\nEvaluated & (prob>=0.178): {evaluated_and_high}/"
          f"{len(OLD_PAPER_YES)}")
    print(f"Evaluated & below: {evaluated_and_low}/{len(OLD_PAPER_YES)}")
    print(f"Not evaluated (pair absent from tested universe): {not_evaluated}/{len(OLD_PAPER_YES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
