import torch


def standardize_columns(values: torch.Tensor) -> torch.Tensor:
    means = values.mean(dim=0)
    stds = values.std(dim=0).clamp_min(1e-8)
    return (values - means) / stds


def add_channel_bias(images: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return images + bias.reshape(1, -1, 1, 1)
