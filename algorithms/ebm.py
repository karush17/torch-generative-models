"""Implements the EBM learner."""

from typing import Any

import abc
import torch
import torch.nn as nn
import torch.autograd as ag

from models.convnet_small import ConvNet
from utils.data_utils import grad_norm, RandomCrop
from algorithms.base import BaseTrainer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class EBMTrainer(BaseTrainer, metaclass=abc.ABCMeta):
    """Implements the EBM trainer."""
    def __init__(self, args: Any, env: str, buffer: Any, filename: str):
        """Initializes the EBM learner."""
        self.args = args
        self.batch_size = self.args.batch_size
        self.p_weight = self.args.p_weight
        self.filename = filename
        self.env = env
        self.buffer = buffer
        self.step = 0
        self.state_shape = self.env.reset()['image'].shape
        self.n_channels = self.state_shape[0]
        self.imsize = self.state_shape[1]
        self.im_len = self.n_channels*self.imsize*self.imsize
        self.action_dim = self.env.action_space.low.size
        self.reservoir = torch.FloatTensor(1000, self.n_channels,
                                           self.imsize, self.imsize).uniform_(0, 1)
        self.buffer_size = len(self.reservoir)
        self.ebm = ConvNet(self.args, self.state_shape, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.ebm.parameters(), lr=self.args.lr_ebm, \
                            betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        if self.args.image_augmentation == True:
            self.augmentation_transform = RandomCrop(self.imsize, 4,
                                                     device=device)

    def train(self, batch: Any):
        """Trains the EBM learner."""
        stats = []
        self.step += 1
        states = torch.Tensor(batch['observations']).to(device)
        states = states.view(states.shape[0], self.n_channels,
                             self.imsize, self.imsize)
        if self.args.image_augmentation == True:
            states = self.augmentation_transform(states)
        self.decay_lr()
        samples = self.sample()
        p_states = self.ebm(states).mean()
        p_samples = self.ebm(samples).mean()
        loss = - self.p_weight*(p_states - p_samples)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ebm.parameters(), self.args.clip)
        self.optimizer.step()
        if loss.abs().item() > 1e8:
            print('ebm diverged')
        stats = {'loss': loss.item(), 'norm': grad_norm(self.ebm)}
        return stats

    def sample_persistent(self):
        """Persistent MCMC sampling with a reservoir."""
        random_samples = torch.FloatTensor(self.batch_size,
                                           self.n_channels, self.imsize,
                                           self.imsize).uniform_(0, 1).to(device)
        if len(self.reservoir) == 0:
            return random_samples
        inds = torch.randint(0, self.buffer_size, (self.batch_size,))
        buffer_samples = self.reservoir[inds].to(device)
        rands = (torch.rand(self.batch_size) < \
                 self.args.reinit_freq).float()[:, None, None, None].to(device)
        samples = rands*random_samples + (1-rands)*buffer_samples
        return samples, inds

    def sample(self, init_samples: list = [], sgld_steps: int = None):
        """Samples from the EBM."""
        self.ebm.eval()
        if init_samples==[]:
            init_samples, inds = self.sample_persistent()
        else:
            init_samples = torch.Tensor(init_samples).to(device)
            init_samples = init_samples.view(init_samples.shape[0],
                                             self.n_channels,
                                             self.imsize, self.imsize)
        if sgld_steps==None:
            sgld_steps = self.args.sgld_steps
        x_k = ag.Variable(init_samples, requires_grad=True)
        for k in range(sgld_steps):
            grads = ag.grad(self.ebm(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += self.args.lr_sgld*grads + \
                self.args.sgld_std*torch.randn_like(x_k)
            x_k = torch.clamp(x_k, 0, 1)
        self.ebm.train()
        final_samples = x_k.detach()
        if len(self.reservoir) > 0 and sgld_steps == None:
            self.reservoir[inds] = final_samples.cpu()
        return final_samples

    def decay_lr(self):
        """Decay learning rate."""
        if self.step in self.args.decay_epochs:
            for param_group in self.optimizer.param_groups:
                new_lr = param_group['lr'] * self.args.decay_rate
                param_group['lr'] = new_lr

    def save(self):
        """Save the EBM model."""
        torch.save(self.ebm.state_dict(), self.filename+'model.pt')

    def load(self):
        """Load the EBM model."""
        self.ebm.load_state_dict(torch.load(self.filename))
