import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, args, state_shape, action_dims):
        super(Generator, self).__init__()

        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.init_size = self.imsize // 4
        self.h_dim = 100
        self.l1 = nn.Linear(self.h_dim, 32 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32, 0.8)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16, 0.8)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(16, self.n_channels, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        out = self.l1(state)
        out = out.view(out.shape[0], 32, self.init_size, self.init_size)
        h = self.bn1(out)
        h = self.up1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu1(h)
        h = self.up2(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.tanh(h)
        return h

class Discriminator(nn.Module):
    def __init__(self, args, state_shape, action_dims):
        super(Discriminator, self).__init__()

        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.conv1 = nn.Conv2d(self.n_channels, 8, 3, 2, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.drop3 = nn.Dropout2d(0.25)
        ds_size = self.imsize // 2 ** 3
        self.l1 = nn.Linear(32*ds_size ** 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        h = self.conv1(state)
        h = self.relu1(h)
        h = self.drop1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.drop2(h)
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.drop3(h)
        out = h.view(h.shape[0], -1)
        out = self.l1(out)
        return self.sigmoid(out)

