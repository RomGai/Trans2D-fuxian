import torch
import torch.nn as nn

from .attention2d import Attention2D


class Trans2DBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention2D(d_model, num_heads, num_channels, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, D]
        attn_out = self.attn(x, padding_mask, causal_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
