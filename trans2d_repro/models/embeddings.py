from typing import List

import torch
import torch.nn as nn


class AttributeEmbedding(nn.Module):
    """Embeds multi-attribute categorical input [B, N, C] -> [B, N, C, d]."""

    def __init__(self, vocab_sizes: List[int], d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_channels = len(vocab_sizes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vs, d_model, padding_idx=0) for vs in vocab_sizes]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        embs = []
        for c, emb in enumerate(self.embeddings):
            # x[:, :, c]: [B, N] -> [B, N, d]
            embs.append(emb(x[:, :, c]))
        # [B, N, C, d]
        out = torch.stack(embs, dim=2)
        return self.dropout(out)
