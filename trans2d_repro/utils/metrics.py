from typing import Dict, Iterable, List

import numpy as np
import torch


def _dcg(binary_relevance: np.ndarray) -> float:
    if binary_relevance.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, 2 + binary_relevance.size))
    return float((binary_relevance / denom).sum())


def ranking_metrics_grouped(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    ks: Iterable[int] = (1, 2, 5),
) -> Dict[str, float]:
    """Compute Precision@k, Hit@k, NDCG@k within each candidate group."""
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    groups_np = group_ids.detach().cpu().numpy()
    ks = list(ks)

    out: Dict[str, List[float]] = {f"Precision@{k}": [] for k in ks}
    out.update({f"Hit@{k}": [] for k in ks})
    out.update({f"NDCG@{k}": [] for k in ks})

    for gid in np.unique(groups_np):
        idx = np.where(groups_np == gid)[0]
        if idx.size == 0:
            continue

        group_scores = logits_np[idx]
        group_labels = labels_np[idx]
        order = np.argsort(-group_scores)
        sorted_labels = group_labels[order]

        positives = int(group_labels.sum())
        for k in ks:
            topk = sorted_labels[:k]
            tp = int(topk.sum())
            out[f"Precision@{k}"].append(tp / float(k))
            out[f"Hit@{k}"].append(1.0 if tp > 0 else 0.0)

            dcg_k = _dcg(topk)
            ideal = np.sort(group_labels)[::-1][:k]
            idcg_k = _dcg(ideal)
            ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0
            out[f"NDCG@{k}"].append(float(ndcg_k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}
