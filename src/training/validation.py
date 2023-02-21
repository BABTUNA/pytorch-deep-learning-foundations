import torch


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            total_loss += loss_fn(logits, targets).item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
    size = len(loader.dataset)
    return {"loss": total_loss / size, "accuracy": correct / size}
