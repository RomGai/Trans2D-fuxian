from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
