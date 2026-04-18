from typing import List

import torch
import torch.nn as nn

from .embeddings import AttributeEmbedding
from .trans2d_block import Trans2DBlock


class Trans2DModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int = 16,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_channels = len(vocab_sizes)
        self.embedding = AttributeEmbedding(vocab_sizes, d_model, dropout)
        self.blocks = nn.ModuleList(
            [Trans2DBlock(d_model, num_heads, self.num_channels, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        attributes: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        attributes: [B, N, C]
        attention_mask: [B, N] bool
        candidate_positions: [B] long
        returns logits: [B]
        """
        b, n, _ = attributes.shape
        x = self.embedding(attributes)  # [B, N, C, D]

        causal_mask = torch.tril(torch.ones((n, n), dtype=torch.bool, device=attributes.device))
        for block in self.blocks:
            x = block(x, attention_mask, causal_mask)

        # gather candidate transformed representation from each sample
        batch_idx = torch.arange(b, device=attributes.device)
        candidate_repr = x[batch_idx, candidate_positions]  # [B, C, D]
        pooled = candidate_repr.mean(dim=1)  # [B, D]
        logits = self.head(pooled).squeeze(-1)  # [B]
        return logits
