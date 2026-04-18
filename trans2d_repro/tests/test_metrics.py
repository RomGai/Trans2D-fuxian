import torch

from utils.metrics import ranking_metrics_grouped


def test_grouped_metrics_simple_case():
    # group 0: correct item ranked first
    # group 1: correct item ranked second
    logits = torch.tensor([0.9, 0.1, 0.2, 0.8])
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    groups = torch.tensor([0, 0, 1, 1])

    m = ranking_metrics_grouped(logits, labels, groups, ks=[1, 2])
    assert abs(m["Hit@1"] - 0.5) < 1e-6
    assert abs(m["Hit@2"] - 1.0) < 1e-6
    assert "NDCG@2" in m
