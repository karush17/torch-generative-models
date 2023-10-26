"""Implements the convnet model."""

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvNet(nn.Module):
    """Implements the deep convolutional model.
    
    Attributes:
        args: parse arguments.
        n_channels: number of channels.
        imsize: image size.
        h_dim: number of hidden dimensions.
        conv: convolutional layer.
        relu: relu activation layer.
        flatten: flatten layer.
        linear: linear MLP layer.
    """
    def __init__(self, args: Any, state_shape: Tuple[int, int],
                 action_dims: int):
        """Initializes the conv model object."""
        super(ConvNet, self).__init__()

        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.h_dim = 128*3*3
        self.action_dims = action_dims
        self.conv11 = nn.Conv2d(self.n_channels, 16, kernel_size=3,
                                stride=1, padding=1)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = Flatten()

        self.linear1 = nn.Linear(self.h_dim, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def encode(self, state: torch.Tensor):
        """Encodes the input image."""
        h = self.conv11(state)
        h = self.conv12(h)
        h = lip_swish(h)
        h = self.conv21(h)
        h = self.conv22(h)
        h = lip_swish(h)
        h = self.conv31(h)
        h = self.conv32(h)
        h = lip_swish(h)
        h = self.conv4(h)
        h = lip_swish(h)
        h = F.avg_pool2d(h, 2)
        return self.flatten(h)

    def forward(self, state: torch.Tensor):
        """Executes the forward pass."""
        h = self.encode(state)
        x = lip_swish(self.linear1(h))
        x = lip_swish(self.linear2(x))
        x = lip_swish(self.linear3(x))
        out = self.linear4(x)
        return out.squeeze()
