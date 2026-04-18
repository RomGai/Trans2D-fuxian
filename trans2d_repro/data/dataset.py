from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .preprocessing import build_expanded_samples, read_snapshot_jsonl


class WatchlistRankingDataset(Dataset):
    """Candidate-expanded dataset for watchlist ranking."""

    def __init__(
        self,
        path: str,
        attribute_names: List[str],
        hash_mod: Dict[str, int],
        max_seq_len: int,
    ) -> None:
        super().__init__()
        records = read_snapshot_jsonl(path, attribute_names, hash_mod)
        self.samples = build_expanded_samples(records, attribute_names, max_seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # attributes: [N, C]
        return {
            "attributes": torch.tensor(s.attributes, dtype=torch.long),
            "label": torch.tensor(s.label, dtype=torch.float32),
            "group_id": torch.tensor(s.group_id, dtype=torch.long),
            "candidate_position": torch.tensor(s.candidate_position, dtype=torch.long),
        }
