"""Implements the EBMVAE model."""

from typing import Any

import abc
import torch
import torch.autograd as ag
import torch.nn.functional as F

from models.vae import VAE
from utils.data_utils import RandomCrop
from algorithms.base import BaseTrainer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class EBMVAETrainer(BaseTrainer, metaclass=abc.ABCMeta):
    """Implements the EBMVAE trainer."""
    def __init__(self, args: Any, env: str, buffer: Any, filename: str):
        """Initializes the learner object."""
        self.args = args
        self.batch_size = self.args.batch_size
        self.p_weight = self.args.p_weight
        self.filename = filename
        self.env = env
        self.buffer = buffer
        self.step = 0
        self.vae_step = 0
        self.state_shape = self.env.reset()['image'].shape
        self.n_channels = self.state_shape[0]
        self.imsize = self.state_shape[1]
        self.im_len = self.n_channels*self.imsize*self.imsize
        self.action_dim = self.env.action_space.low.size
        self.reservoir = torch.FloatTensor(1000, self.n_channels,
                                           self.imsize,
                                           self.imsize).uniform_(0, 1)
        self.latent = torch.FloatTensor(4000, 2048).uniform_(-2, 2)
        self.buffer_size = len(self.reservoir)
        self.vae = VAE(self.args, self.state_shape, self.action_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(),
                                            lr=self.args.lr,
                                            betas=[0.9, 0.999],
                                            weight_decay=self.args.weight_decay)
        if self.args.image_augmentation == True:
            self.augmentation_transform = RandomCrop(self.imsize, 4,
                                                     device=device)

    def train(self, batch: Any):
        """Trains the learner."""
        stats = []
        self.step += 1
        states = torch.Tensor(batch['observations']).to(device)
        states = states.view(states.shape[0],
                             self.n_channels,
                             self.imsize,
                             self.imsize)
        if self.args.image_augmentation == True:
            states = self.augmentation_transform(states)
        self.decay_lr()
        recon_x, mu, logvar = self.vae(states)
        bce = F.binary_cross_entropy(recon_x, states).mean()
        kl = -0.5*torch.mean(1+logvar - mu.pow(2) - logvar.exp())
        loss = bce + 0.005*kl

        samples = self.sample()
        p_states = self.vae(states)[1].mean()
        p_samples = self.vae(samples)[1].mean()
        ebm_loss = - self.p_weight*(p_states - p_samples)
        loss += 0.1*ebm_loss
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        if loss.abs().item() > 1e8:
            print('ebm diverged')
        stats = {'loss': loss.item()}
        return stats

    def sample_persistent(self):
        """Persistent MCMC sampling from the reservoir."""
        random_samples = torch.FloatTensor(self.batch_size,
                                           self.n_channels,
                                           self.imsize, 
                                           elf.imsize).uniform_(0, 1).to(device)
        if len(self.reservoir) == 0:
            return random_samples
        inds = torch.randint(0, self.buffer_size, (self.batch_size,))
        buffer_samples = self.reservoir[inds].to(device)
        rands = (torch.rand(self.batch_size) <\
                  self.args.reinit_freq).float()[:, None, None, None].to(device)
        samples = rands*random_samples + (1-rands)*buffer_samples
        return samples, inds

    def sample_vae(self, init_samples: list = [], sgld_steps: int = None):
        """Sample from the VAE."""
        self.vae.eval()
        if init_samples==[]:
            init_samples, _ = self.sample_persistent()
        else:
            init_samples = torch.Tensor(init_samples).to(device)
            init_samples = init_samples.view(init_samples.shape[0],
                                             self.n_channels,
                                             self.imsize, self.imsize)
        final_samples, _, _ = self.vae(init_samples)
        self.vae.train()
        return final_samples.detach()

    def sample(self, init_samples: list = [], sgld_steps: int = None):
        """Sample from the EBMVAE model."""
        self.vae.eval()
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
            x_k, _, _ = self.vae(x_k)
            grads = ag.grad(self.vae(x_k)[1].sum(), [x_k], retain_graph=True)[0]
            x_k.data += self.args.lr_sgld*grads +\
                  self.args.sgld_std*torch.randn_like(x_k)
            x_k = torch.clamp(x_k, 0, 1)
        self.vae.train()
        final_samples = x_k.detach()
        if len(self.reservoir) > 0 and sgld_steps==None:
            self.reservoir[inds] = final_samples.cpu()
        return final_samples

    def decay_lr(self):
        """Decay the learning rate."""
        if self.step in self.args.decay_epochs:
            for param_group in self.vae_optimizer.param_groups:
                new_lr = param_group['lr'] * self.args.decay_rate
                param_group['lr'] = new_lr

    def save(self):
        """Save the model."""
        torch.save(self.vae.state_dict(), self.filename+'model.pt')

    def load(self):
        """Load the model."""
        self.vae.load_state_dict(torch.load(self.filename))
