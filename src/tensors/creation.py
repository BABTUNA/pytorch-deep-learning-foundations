import torch


def demo_tensors():
    scalar = torch.tensor(7)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.zeros((2, 3))
    random_batch = torch.rand((4, 3))
    return scalar, vector, matrix, random_batch
