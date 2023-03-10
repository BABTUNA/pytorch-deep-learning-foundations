from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def make_scheduler(name, optimizer):
    if name == "step":
        return StepLR(optimizer, step_size=8, gamma=0.5)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2)
    raise ValueError(f"unknown scheduler: {name}")
