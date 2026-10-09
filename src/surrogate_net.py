"""The network used by the POD surrogate (shared by training and the inverse problem)."""
import torch.nn as nn


def make_network(n_out, width=128):
    """Inputs: Re, cos(phase), sin(phase).  Outputs: n_out POD coefficients."""
    return nn.Sequential(
        nn.Linear(3, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, n_out),
    )
