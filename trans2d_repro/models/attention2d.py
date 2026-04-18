import math

import torch
import torch.nn as nn

from .linear2d import Linear2D


class Attention2D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear2D(num_channels, d_model, d_model)
        self.k_proj = Linear2D(num_channels, d_model, d_model)
        self.v_proj = Linear2D(num_channels, d_model, d_model)
        self.out_proj = Linear2D(num_channels, d_model, d_model)

        self.alpha_full = nn.Parameter(torch.tensor(1.0))
        self.alpha_item = nn.Parameter(torch.tensor(1.0))
        self.alpha_channel = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, D] -> [B, H, N, C, Dh]
        b, n, c, _ = x.shape
        x = x.view(b, n, c, self.num_heads, self.d_head)
        return x.permute(0, 3, 1, 2, 4).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, C, Dh] -> [B, N, C, D]
        b, h, n, c, dh = x.shape
        return x.permute(0, 2, 3, 1, 4).contiguous().view(b, n, c, h * dh)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C, D]
        padding_mask: [B, N] bool
        causal_mask: [N, N] bool, True for allowed (j <= i)
        """
        b, n, c, _ = x.shape

        q = self._split_heads(self.q_proj(x))  # [B, H, N, C, Dh]
        k = self._split_heads(self.k_proj(x))  # [B, H, N, C, Dh]
        v = self._split_heads(self.v_proj(x))  # [B, H, N, C, Dh]

        # Full attention A_F over item-attribute pairs: [B, H, N, C, N, C]
        full_logits = torch.einsum("bhncd,bhmkd->bhncmk", q, k)

        # Item/time attention A_I: [B, H, N, N] then broadcast to [B,H,N,C,N,C]
        q_item = q.mean(dim=3)  # [B, H, N, Dh]
        k_item = k.mean(dim=3)  # [B, H, N, Dh]
        item_logits = torch.einsum("bhnd,bhmd->bhnm", q_item, k_item)
        item_logits = item_logits.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, c, -1, c)

        # Channel/feature attention A_C: [B, H, C, C] then broadcast
        q_ch = q.mean(dim=2)  # [B, H, C, Dh]
        k_ch = k.mean(dim=2)  # [B, H, C, Dh]
        ch_logits = torch.einsum("bhcd,bhkd->bhck", q_ch, k_ch)
        ch_logits = ch_logits.unsqueeze(2).unsqueeze(4).expand(-1, -1, n, -1, n, -1)

        logits = (
            self.alpha_full * full_logits
            + self.alpha_item * item_logits
            + self.alpha_channel * ch_logits
        ) / math.sqrt(self.d_head)

        # Causal mask on item dimension (keys cannot be future positions)
        # allowed_item: [1,1,N,1,N,1]
        allowed_item = causal_mask.view(1, 1, n, 1, n, 1)
        logits = logits.masked_fill(~allowed_item, float("-inf"))

        # Padding key mask: [B,1,1,1,N,1]
        key_valid = padding_mask.view(b, 1, 1, 1, n, 1)
        logits = logits.masked_fill(~key_valid, float("-inf"))

        # Softmax over flattened key dimensions (N*C)
        logits_flat = logits.view(b, self.num_heads, n, c, n * c)
        attn = torch.softmax(logits_flat, dim=-1)
        attn = self.dropout(attn)

        # Flatten values over key dims: [B,H,N*C,Dh]
        v_flat = v.reshape(b, self.num_heads, n * c, self.d_head)
        # Output: [B,H,N,C,Dh]
        out = torch.einsum("bhncf,bhfd->bhncd", attn, v_flat)

        # Zero padded query positions
        query_valid = padding_mask.view(b, 1, n, 1, 1)
        out = out * query_valid

        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out
