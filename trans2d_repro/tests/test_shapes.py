import torch

from models.linear2d import Linear2D
from models.trans2d import Trans2DModel


def test_linear2d_shape():
    layer = Linear2D(num_channels=6, in_dim=8, out_dim=16)
    x = torch.randn(4, 7, 6, 8)
    y = layer(x)
    assert y.shape == (4, 7, 6, 16)


def test_candidate_scoring_shape():
    model = Trans2DModel(vocab_sizes=[100] * 5, d_model=16, num_heads=4, num_layers=1, dropout=0.0)
    attrs = torch.randint(1, 99, (3, 9, 5))
    mask = torch.ones(3, 9, dtype=torch.bool)
    cand_pos = torch.tensor([8, 8, 8], dtype=torch.long)
    logits = model(attrs, mask, cand_pos)
    assert logits.shape == (3,)
