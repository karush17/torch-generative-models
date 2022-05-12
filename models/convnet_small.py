import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def lip_swish(x):
    return (x*torch.sigmoid(x)) / 1.1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 1, 1)

class ConvNet(nn.Module):
    def __init__(self, args, state_shape, action_dims):
        super(ConvNet, self).__init__()
        
        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.h_dim = 16*48*48
        self.action_dims = action_dims
        self.conv11 = nn.Conv2d(self.n_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = Flatten()

        self.linear1 = nn.Linear(self.h_dim, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def encode(self, state):
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
        return self.flatten(h)

    def forward(self, state):
        h = self.encode(state)
        x = F.relu(self.linear1(h))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        out = self.linear4(x)
        return out.squeeze()

