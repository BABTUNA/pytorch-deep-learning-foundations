from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics},
        path,
    )


def load_weights(path, model, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
