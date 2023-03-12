from torch import nn


def conv_block(input_channels, output_channels, dropout=0.1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.Dropout2d(dropout),
        nn.MaxPool2d(2),
    )
