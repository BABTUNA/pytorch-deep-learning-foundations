import torch


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    lengths = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    return values / lengths.clamp_min(1e-8)


def pairwise_scores(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right.T
