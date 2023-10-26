"""Implements the score network model."""

from typing import Any, Tuple

import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def lip_swish(x: torch.Tensor):
    """Implements the lip swish activation."""
    return (x * torch.sigmoid(x)) / 1.1

class Flatten(nn.Module):
    """Flattens the input tensor."""

    def forward(self, input: torch.Tensor):
        """Executes the forward method."""
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    """Unflattens the input tensor."""

    def forward(self, input: torch.Tensor, size: int = 128):
        """Executes the forward method."""
        return input.view(input.size(0), size, 1, 1)

class ScoreConvNet(nn.Module):
    """Implements the Score function architecture.
    
    Attributes:
        args: parser arguments.
        n_channels: number of channels.
        imsize: size of the imnput image.
        h_dim: number of hidden dimensions.
        action_dims: number of actions.
        conv: convolutional layer.
        norm: normalization technique.
        elu: elu activation layer.
        linear: fully connected mlp layer.
        flatten: flattening layer.
    """
    def __init__(self, args: Any, state_shape: Tuple[int, int],
                 action_dims: int):
        """Initializes the score model object."""
        super(ScoreConvNet, self).__init__()

        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.h_dim = 3*48*48
        self.action_dims = action_dims
        self.conv1 = nn.Conv2d(self.n_channels, 16,
                               kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(4, 16)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(4, 32)
        self.elu2 = nn.ELU()
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3,
                                        stride=1, padding=1)
        self.norm3 = nn.GroupNorm(4, 16)
        self.elu3 = nn.ELU()
        self.conv4 = nn.ConvTranspose2d(16, self.n_channels,
                                        kernel_size=4, stride=2, padding=1)
        self.elu4 = nn.ELU()

        self.linear1 = nn.Linear(self.h_dim, 512)
        self.norm4 = nn.LayerNorm(512)
        self.elu5 = nn.ELU()
        self.linear2 = nn.Linear(512, self.h_dim)
        self.flatten = Flatten()

    def forward(self, state: torch.Tensor):
        """Executes the forward pass."""
        h = self.conv1(state)
        h = self.norm1(h)
        h = self.elu1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.elu2(h)
        h = self.conv3(h)
        h = self.norm3(h)
        h = self.elu3(h)
        h = self.conv4(h)
        h = self.elu4(h)
        h = self.flatten(h)
        h = self.linear1(h)
        h = self.norm4(h)
        h = self.elu5(h)
        h = self.linear2(h)
        return h.view(h.shape[0], self.n_channels, self.imsize, self.imsize)
