import torch


def predict(model, inputs, device="cpu"):
    model.eval()
    with torch.inference_mode():
        logits = model(inputs.to(device))
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.cpu()
