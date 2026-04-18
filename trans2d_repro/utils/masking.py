import torch


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
