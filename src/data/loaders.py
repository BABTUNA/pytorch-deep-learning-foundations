from torch.utils.data import DataLoader


def make_loaders(training, validation, batch_size=32):
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader
