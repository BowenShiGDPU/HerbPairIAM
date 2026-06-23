"""HerbEmbIAM — learnable-embedding ablation of HerbPairIAM.

Design
------
Same architecture as ``InteractionAwareSetModel`` **except** the KG-derived
individual herb profile and the KG-derived ADR profile are replaced by
learnable embedding tables::

    herb_emb : nn.Embedding(n_herbs, d)
    adr_emb  : nn.Embedding(n_adrs,  d)

The pair branch keeps its KG-informed features (the 11 columns of
``PAIR_FEATURE_KEYS``) so this is a *minimal* learnable-embedding
substitution: individual level switches from KG-SVD to learnable;
pair level unchanged; dose branch absent. See ``herb_pair_iam.py`` for
why the primary model uses KG features.

Purpose (the paper claim this ablation establishes)
---------------------------------------------------
Nature Communications reviewers routinely ask whether learnable
representations could replace hand-crafted KG features. With only 707
positive labels spread over 126 herbs × 963 ADRs (mean < 10⁻³
observations per (herb, ADR) pair), learnable identity-level embeddings
are expected to underfit. If this ablation does underperform HerbPairIAM,
we establish the KG-feature framework as *necessary* rather than
convenient in the low-data regime. If it matches, we revise the paper.

Fair-comparison notes
---------------------
* Embedding dimension ``d=48`` matches the KG profile dimension
  (``PROFILE_DIM``) used by HerbPairIAM. Parameter count is therefore
  dominated by the herb / ADR tables rather than by the set operator.
* Hyperparameters (hidden=32, dropout=0.3, lr=1e-3, epochs=100,
  patience=10, batch_size=32, neg_ratio=10, val_metric=AUROC) are
  inherited from HerbPairIAM to keep the comparison at fixed training
  budget.
* Training consumes the same per-fold (train_idx, val_idx, test_idx)
  partitions as the canonical 3-seed head-to-head, so the 30-fold
  comparison is paired at the fold level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HerbEmbIAM(nn.Module):
    def __init__(
        self,
        herb_vocab: list[str],
        adr_vocab: list[str],
        pair_in: int,
        hidden: int = 32,
        emb_dim: int = 48,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.herb_to_idx: dict[str, int] = {h: i for i, h in enumerate(herb_vocab)}
        self.adr_to_idx: dict[str, int] = {a: i for i, a in enumerate(adr_vocab)}
        # Padding index 0 is intentionally reserved for unseen herbs/ADRs so
        # the model degrades gracefully on cold-start rather than crashing.
        n_herbs = len(herb_vocab) + 1
        n_adrs = len(adr_vocab) + 1
        self.herb_emb = nn.Embedding(n_herbs, emb_dim, padding_idx=0)
        self.adr_emb = nn.Embedding(n_adrs, emb_dim, padding_idx=0)
        # Standard normal init with modest scale so attention logits stay in
        # range; matches the effective range of the KG-SVD profile after
        # projection through ``node_enc``.
        nn.init.normal_(self.herb_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.adr_emb.weight, mean=0.0, std=0.1)
        self.herb_emb.weight.data[0].zero_()
        self.adr_emb.weight.data[0].zero_()

        # After embedding, the individual node feature is [emb_dim] with no
        # KG extras. The learnable node_enc converts it to the hidden size
        # used by the rest of the network.
        self.node_enc = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # Pair branch identical to IAM.
        self.pair_enc = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # ADR encoder: project the learnable adr embedding down to ``hidden``.
        self.adr_enc = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.node_attn = nn.Linear(hidden * 2, 1)
        self.pair_attn = nn.Linear(hidden * 2, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _herb_indices(self, herbs: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.herb_to_idx.get(h, 0) for h in herbs],
            dtype=torch.long,
            device=device,
        )

    def _adr_index(self, adr_id: str, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.adr_to_idx.get(adr_id, 0)],
            dtype=torch.long,
            device=device,
        )

    def forward(self, sample: dict):
        # We rely on the raw (string) herb ids and adr id stored in the
        # sample dict by ``neural_models.precompute_samples``. This avoids
        # changes to the shared sample builder.
        herbs = sample.get("herbs")
        adr_id = sample.get("adr_id")
        device = sample["pair_features"].device

        herb_ix = self._herb_indices(list(herbs), device)
        node_h = self.node_enc(self.herb_emb(herb_ix))  # [n_herbs, hidden]
        a_vec = self.adr_enc(self.adr_emb(self._adr_index(adr_id, device)).squeeze(0))  # [hidden]

        a_exp = a_vec.unsqueeze(0).expand(node_h.size(0), -1)
        node_alpha = F.softmax(
            self.node_attn(torch.cat([node_h, a_exp], dim=-1)).squeeze(-1),
            dim=0,
        )
        z_node = (node_alpha.unsqueeze(-1) * node_h).sum(dim=0)

        pair_tensor = sample["pair_features"]
        if pair_tensor.shape[0] > 0:
            pair_h = self.pair_enc(pair_tensor)
            a_pair = a_vec.unsqueeze(0).expand(pair_h.size(0), -1)
            pair_alpha = F.softmax(
                self.pair_attn(torch.cat([pair_h, a_pair], dim=-1)).squeeze(-1),
                dim=0,
            )
            z_pair = (pair_alpha.unsqueeze(-1) * pair_h).sum(dim=0)
        else:
            pair_alpha = torch.zeros((0,), device=device)
            z_pair = torch.zeros_like(z_node)

        logit = self.pred(torch.cat([z_node, z_pair, a_vec], dim=-1)).squeeze(-1)
        return logit, {"herb_attn": node_alpha.detach(), "pair_attn": pair_alpha.detach()}


MODEL_NAME = "HerbEmbIAM"
