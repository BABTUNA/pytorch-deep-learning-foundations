import torch

from src.tensors.broadcasting import standardize_columns


def test_standardize_columns_keeps_shape():
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    result = standardize_columns(values)
    assert result.shape == values.shape
    assert torch.allclose(result.mean(dim=0), torch.zeros(2), atol=1e-6)
