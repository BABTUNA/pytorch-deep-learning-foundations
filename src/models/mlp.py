from torch import nn


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, classes),
        )

    def forward(self, inputs):
        return self.layers(inputs)
