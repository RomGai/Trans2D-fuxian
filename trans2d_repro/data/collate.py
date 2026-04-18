from typing import Dict, List

import torch


def collate_watchlist_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Returns:
      attributes: [B, N_max, C]
      attention_mask: [B, N_max] bool, True where valid token exists
      labels: [B]
      group_ids: [B]
      candidate_positions: [B]
    """
    bsz = len(batch)
    n_max = max(x["attributes"].shape[0] for x in batch)
    c = batch[0]["attributes"].shape[1]

    attrs = torch.zeros((bsz, n_max, c), dtype=torch.long)
    attn_mask = torch.zeros((bsz, n_max), dtype=torch.bool)
    labels = torch.zeros((bsz,), dtype=torch.float32)
    groups = torch.zeros((bsz,), dtype=torch.long)
    cand_pos = torch.zeros((bsz,), dtype=torch.long)

    for i, item in enumerate(batch):
        n_i = item["attributes"].shape[0]
        attrs[i, :n_i] = item["attributes"]
        attn_mask[i, :n_i] = True
        labels[i] = item["label"]
        groups[i] = item["group_id"]
        cand_pos[i] = item["candidate_position"]

    return {
        "attributes": attrs,
        "attention_mask": attn_mask,
        "labels": labels,
        "group_ids": groups,
        "candidate_positions": cand_pos,
    }
