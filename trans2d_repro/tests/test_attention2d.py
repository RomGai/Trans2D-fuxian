import torch

from models.attention2d import Attention2D


def test_attention2d_mask_behavior_zeroes_padded_queries():
    attn = Attention2D(d_model=16, num_heads=4, num_channels=3, dropout=0.0)
    x = torch.randn(2, 5, 3, 16)
    padding_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    causal = torch.tril(torch.ones((5, 5), dtype=torch.bool))

    y = attn(x, padding_mask, causal)
    assert y.shape == (2, 5, 3, 16)
    # Query positions 3,4 are padded in sample 2 -> should be all zeros after masking.
    assert torch.allclose(y[1, 3:], torch.zeros_like(y[1, 3:]), atol=1e-6)
