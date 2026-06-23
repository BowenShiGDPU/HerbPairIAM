"""Run the HerbEmbIAM learnable-embedding ablation.

Trains HerbEmbIAM (KG individual features replaced by learnable
``nn.Embedding`` tables) under the same 3-seed protocol as the primary
head-to-head (``run_dose_head2head.py``). Writes fold pickles into the
``dose_head2head/fold_results/`` subdirectory so the downstream
aggregation scripts can pool HerbEmbIAM alongside HerbPairIAM /
InteractionAwareSetModel / DoseAwareIAM with no additional wiring.

Naming: ``H2H_seed<S>_HerbEmbIAM_fold<K>.pkl``.

Usage::

    RESULTS_ROOT_DIR=results \\
    EXPERIMENT_SUBDIR=formal_doseaware_neg10_auroc/dose_head2head \\
    VAL_SELECTION_METRIC=auroc \\
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    python -u src/scripts/run_herb_emb_iam.py --seeds 42 13 7 --neg-ratio 10
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_SRC = _pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import argparse
import sys

from data.phase2_dataset import INNER_VAL_FRAC, OUTER_FOLDS
from experiment_utils import FOLD_RESULTS_DIR, load_pickle, sanitize_name, save_pickle
from models.neural_models import (
    ModelConfig,
    build_sample_collections,
    summarize_results,
    train_one_split,
)
from phase4_evaluation import prepare_common_inputs


sys.stdout.reconfigure(line_buffering=True)


def _build_fold_splits(df, outer_seed: int) -> list[dict]:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    splits = []
    for fold_id, (train_val_idx, test_idx) in enumerate(
        outer.split(df.index.values, df["label"].values)
    ):
        train_val_idx = np.asarray(train_val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_FRAC,
            random_state=outer_seed + fold_id,
        )
        inner_train, inner_val = next(
            inner.split(train_val_idx, df.iloc[train_val_idx]["label"].values)
        )
        train_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]
        splits.append(
            {
                "fold": fold_id,
                "outer_seed": outer_seed,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }
        )
    return splits


def _vocab_from_lookups(lookups: dict, df) -> dict:
    """Build the (herb, ADR) vocabulary from the dataset lookups + dataframe.

    Herbs live in ``lookups['h2i']`` (herb id -> list of ingredient ids).
    ADRs are whatever appears in ``df['Adr_id']``. We sort both lists so
    the integer mapping is deterministic.
    """
    herb_vocab = sorted(lookups.get("h2i", {}).keys())
    adr_vocab = sorted(df["Adr_id"].unique().tolist())
    return {"herb_vocab": herb_vocab, "adr_vocab": adr_vocab}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7])
    parser.add_argument("--neg-ratio", type=int, default=10)
    parser.add_argument("--fold", type=int, nargs="+", default=None)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    ds, df, _, _, labels, hp, ap, pf, lookups = prepare_common_inputs()
    # HerbEmbIAM uses *baseline* samples (no dose tails), matching
    # InteractionAwareSetModel. That way the only axis of variation vs.
    # HerbPairIAM/IAM on the same seed is the source of individual herb
    # and ADR representation (KG-SVD vs learnable embedding).
    sample_map = build_sample_collections(
        df, lookups, hp, ap, pf, ["InteractionAwareSetModel"]
    )
    samples = sample_map["InteractionAwareSetModel"]
    vocab = _vocab_from_lookups(lookups, df)
    print(
        f"[herb_emb_iam] vocab: n_herbs={len(vocab['herb_vocab'])}, n_adrs={len(vocab['adr_vocab'])}",
        flush=True,
    )

    cfg = ModelConfig(
        hidden=32, dropout=0.3, lr=1e-3, epochs=100, patience=10,
        batch_size=32, neg_ratio=args.neg_ratio, eval_every=2,
    )

    for seed in args.seeds:
        splits = _build_fold_splits(df, seed)
        if args.fold is not None:
            allowed = set(int(f) for f in args.fold)
            splits = [s for s in splits if int(s["fold"]) in allowed]
        print(
            f"[herb_emb_iam] seed={seed} folds={[int(s['fold']) for s in splits]}",
            flush=True,
        )
        results = []
        for split in splits:
            fold_id = int(split["fold"])
            tag = f"H2H_seed{seed}_HerbEmbIAM"
            path = FOLD_RESULTS_DIR / f"{sanitize_name(tag)}_fold{fold_id}.pkl"
            if path.exists() and not args.no_skip:
                results.append(load_pickle(path))
                print(f"  [skip] {path.name} exists", flush=True)
                continue
            fold_result = train_one_split(
                "HerbEmbIAM", samples, labels, split, cfg,
                save_model=False, vocab=vocab,
            )
            fold_result["ablation_tag"] = tag
            fold_result["outer_seed"] = seed
            fold_result["model"] = "HerbEmbIAM"
            save_pickle(fold_result, path)
            results.append(fold_result)
            print(
                f"  [fold {fold_id}] AUROC={fold_result['auroc']:.4f} "
                f"AUPRC={fold_result['auprc']:.4f}",
                flush=True,
            )
        summary = summarize_results(results)
        print(
            f"[herb_emb_iam] DONE seed={seed} "
            f"AUROC={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f} "
            f"AUPRC={summary['auprc_mean']:.4f}±{summary['auprc_std']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
