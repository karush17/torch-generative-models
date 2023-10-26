"""Implements the wide resnet model."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """Implements 3x3 convolutional filter."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=True)

def conv_init(m: torch.Tensor):
    """Implements the conv layer initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class Identity(nn.Module):
    """Implements the identity module."""

    def __init__(self, *args, **kwargs):
        """Initializes the identity object."""
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Executes the forward method."""
        return x

class wide_basic(nn.Module):
    """Implements the basic wide resnet block.
    
    Attributes:
        lrelu: leaky relu activation layer.
        bn: batch normalization layer.
        conv: convolutional layer.
        dropout: dropout layer.
        shortcut: highway connection for resnet.
    """
    def __init__(self, in_planes, planes, dropout_rate,
                 stride=1, norm=None, leak=.2):
        """Initializes the basic block."""
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 \
            else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=True),
            )

    def forward(self, x: torch.Tensor):
        """Executes the basic block."""
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)
        return out

def get_norm(n_filters: int, norm: Any):
        """Returns the type of normalization."""
        return Identity()

class Wide_ResNet(nn.Module):
    """Implements the wide resnet architecture.
    
    Attributes:
        args: parser arguments.
        n_channels: number of channels.
        imsize: size of the image.
        h_dim: number of hidden dimensions.
        action_dims: number of actions.
        leak: leaky relu constant.
        in_planes: number of in channels.
        sum_pool: sum pool activation.
        norm: normalization type.
        lrelu: leaky relu activation.
    """
    def __init__(self, args, state_shape, action_dims, depth=28, widen_factor=10,
                 sum_pool=False, norm=None, leak=.2, dropout_rate=0.0):
        """Initializes the wide resnet model."""
        super(Wide_ResNet, self).__init__()
        self.args = args
        self.n_channels = state_shape[0]
        self.imsize = state_shape[1]
        self.h_dim = 640
        self.action_dims = action_dims
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(self.n_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n,
                                       dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n,
                                       dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n,
                                       dropout_rate, stride=2)
        self.bn1 = get_norm(nStages[3], self.norm)
        self.linear = nn.Linear(self.h_dim, 1)

    def _wide_layer(self, block: Any, planes: int, num_blocks: int,
                    dropout_rate: float, stride: int):
        """Implements the wide resnet layer by layer."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate,
                                stride, norm=self.norm))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Executes the forward method."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out).squeeze()
