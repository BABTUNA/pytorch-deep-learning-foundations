import torch


def accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def running_average(total, count):
    return total / max(count, 1)
