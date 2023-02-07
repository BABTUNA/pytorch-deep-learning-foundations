import torch


def slope_at(value: float) -> float:
    x = torch.tensor(value, requires_grad=True)
    y = x**3 + 2 * x**2 - x
    y.backward()
    return x.grad.item()
