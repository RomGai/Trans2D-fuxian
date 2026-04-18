from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Candidate:
    attributes: Dict[str, int]
    label: int


@dataclass
class SnapshotRecord:
    user_id: str
    snapshot_id: str
    history: List[Dict[str, int]]
    candidates: List[Candidate]


@dataclass
class ExpandedSample:
    # attributes: [N, C]
    attributes: List[List[int]]
    label: int
    group_id: int
    candidate_position: int
