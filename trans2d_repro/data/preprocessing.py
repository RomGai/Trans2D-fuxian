import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import Candidate, ExpandedSample, SnapshotRecord


def stable_hash_to_range(value: Any, mod: int) -> int:
    """Map arbitrary raw ID to [1, mod] (0 reserved for padding)."""
    raw = str(value).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    return (int(digest, 16) % mod) + 1


def normalize_attributes(
    raw_attrs: Dict[str, Any],
    attribute_names: List[str],
    hash_mod: Dict[str, int],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in attribute_names:
        value = raw_attrs.get(name, 0)
        if name in hash_mod:
            out[name] = stable_hash_to_range(value, hash_mod[name])
        else:
            out[name] = int(value) if value is not None else 0
    return out


def read_snapshot_jsonl(path: str, attribute_names: List[str], hash_mod: Dict[str, int]) -> List[SnapshotRecord]:
    records: List[SnapshotRecord] = []
    src = Path(path)
    if not src.exists():
        return records

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            history = [normalize_attributes(x, attribute_names, hash_mod) for x in row["history"]]
            candidates = [
                Candidate(
                    attributes=normalize_attributes(x["attributes"], attribute_names, hash_mod),
                    label=int(x["label"]),
                )
                for x in row["candidates"]
            ]
            records.append(
                SnapshotRecord(
                    user_id=str(row.get("user_id", "unknown")),
                    snapshot_id=str(row.get("snapshot_id", "0")),
                    history=history,
                    candidates=candidates,
                )
            )
    return records


def build_expanded_samples(
    records: List[SnapshotRecord],
    attribute_names: List[str],
    max_seq_len: int,
) -> List[ExpandedSample]:
    """
    Expand each (snapshot, candidate) into one training sample.
    Sequence format per sample: history + [candidate].
    """
    samples: List[ExpandedSample] = []
    for group_id, record in enumerate(records):
        # history_attrs: list[[C], ...]
        history_attrs = [[itm[name] for name in attribute_names] for itm in record.history]
        for cand in record.candidates:
            candidate_vec = [cand.attributes[name] for name in attribute_names]
            seq = history_attrs + [candidate_vec]
            seq = seq[-max_seq_len:]
            samples.append(
                ExpandedSample(
                    attributes=seq,
                    label=int(cand.label),
                    group_id=group_id,
                    candidate_position=len(seq) - 1,
                )
            )
    return samples
