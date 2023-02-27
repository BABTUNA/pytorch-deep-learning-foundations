from torch.optim import Adam, SGD


def make_optimizer(name, parameters, learning_rate):
    if name == "sgd":
        return SGD(parameters, lr=learning_rate, momentum=0.9)
    if name == "adam":
        return Adam(parameters, lr=learning_rate)
    raise ValueError(f"unknown optimizer: {name}")
