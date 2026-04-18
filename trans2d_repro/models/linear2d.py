import torch
import torch.nn as nn


class Linear2D(nn.Module):
    """Channel-specific linear: each attribute channel has its own W, b."""

    def __init__(self, num_channels: int, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_channels, in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels, out_dim))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, D_in]
        # out[b,n,c,o] = sum_i x[b,n,c,i] * W[c,i,o]
        out = torch.einsum("bnci,cio->bnco", x, self.weight)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(0)
        return out
