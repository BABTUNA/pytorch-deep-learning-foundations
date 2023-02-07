import torch


def linear_loss(features, targets, weights):
    predictions = features @ weights
    return torch.mean((predictions - targets) ** 2)


def one_gradient_step(features, targets, weights, learning_rate=0.01):
    loss = linear_loss(features, targets, weights)
    loss.backward()
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        weights.grad.zero_()
    return loss.item()
