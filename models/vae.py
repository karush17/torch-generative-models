"""Implements the VAE model."""

from typing import Any, Tuple

import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Flatten(nn.Module):
    """Flattens the input tensor."""

    def forward(self, input: torch.Tensor):
        """Executes the flattening method."""
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    """Unflattens the input tensor."""

    def forward(self, input: torch.Tensor, size: int = 128):
        """Executes the forward method."""
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    """Implements the VAE model.
    
    Attributes:
        args: parser arguments.
        n_channels: number of channels.
        imsize: size of the image.
        h_dim: number of hidden dimensions.
        action_dims: number of actions.
        conv: convolutional layer.
        relu: relu activation layer.
        flatten: flattening layer.
        fc: fully connected layer.
        unflatten: unflattening layer.
        convt: conv transpose layer.
        sigmoid: sigmoid activation layer.
    """
    def __init__(self, args: Any, state_shape: Tuple[int, int], 
                 action_dims: int):
        """Initializes the VAE object."""
        super(VAE, self).__init__()

        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.h_dim = 32*12*12
        self.action_dims = action_dims
        self.conv11 = nn.Conv2d(self.n_channels, 16, kernel_size=3,
                                stride=1, padding=1)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv31 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = Flatten()

        self.fc1 = nn.Linear(self.h_dim, 2048)
        self.fc2 = nn.Linear(self.h_dim, 2048)
        self.fc3 = nn.Linear(2048, 128)

        self.unflatten = UnFlatten()
        self.convt1 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2)
        self.relut1 = nn.ReLU()
        self.convt2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.relut2 = nn.ReLU()
        self.convt3 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2)
        self.relut3 = nn.ReLU()
        self.convt4 = nn.ConvTranspose2d(16, self.n_channels, kernel_size=5,
                                         stride=2, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, state: torch.Tensor):
        """Encodes the input image."""
        h = self.conv11(state)
        h = self.conv12(h)
        h = self.relu1(h)
        h = self.conv21(h)
        h = self.conv22(h)
        h = self.relu2(h)
        h = self.conv31(h)
        h = self.conv32(h)
        h = self.relu3(h)
        h = self.conv4(h)
        h = self.relu4(h)
        h = self.flatten(h)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterizes the distribution."""
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h: torch.Tensor):
        """Latent bottleneck."""
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor):
        """Decodes the latent code."""
        z = self.fc3(z)
        z = self.unflatten(z)
        z = self.convt1(z)
        z = self.relut1(z)
        z = self.convt2(z)
        z = self.relut2(z)
        z = self.convt3(z)
        z = self.relut3(z)
        z = self.convt4(z)
        z = self.sigmoid(z)
        return z

    def forward(self, state: torch.Tensor):
        """Executes the forward method."""
        z, mu, logvar = self.encode(state)
        out = self.decode(z)
        return out.squeeze(), mu, logvar
