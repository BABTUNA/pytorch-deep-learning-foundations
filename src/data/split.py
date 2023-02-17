from torch.utils.data import random_split


def train_validation_split(dataset, validation_fraction=0.2, generator=None):
    validation_size = round(len(dataset) * validation_fraction)
    training_size = len(dataset) - validation_size
    return random_split(dataset, [training_size, validation_size], generator=generator)
