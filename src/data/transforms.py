import torch


class Standardize:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, values):
        return (values - self.mean) / self.std.clamp_min(1e-8)
