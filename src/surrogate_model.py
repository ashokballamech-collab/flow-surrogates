"""The neural network used as the cavity-flow surrogate (shared by training and evaluation)."""
import torch.nn as nn


class Surrogate(nn.Module):
    """A small fully connected network: normalized Re (1 number) -> flattened (u, v) field."""

    def __init__(self, n_out, width=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),       # layer 1: 1 input -> 64 neurons
            nn.Linear(width, width), nn.Tanh(),   # layer 2: 64 -> 64 neurons
            nn.Linear(width, n_out),              # output layer: 64 -> 8450 numbers
        )

    def forward(self, x):
        return self.layers(x)
