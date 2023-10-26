"""Implements the VAE learner."""

from typing import Any

import abc
import torch
import torch.nn.functional as F

from models.vae import VAE
from utils.data_utils import RandomCrop
from algorithms.base import BaseTrainer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class VAETrainer(BaseTrainer, metaclass=abc.ABCMeta):
    """Implements the VAE trainer."""
    def __init__(self, args: Any, env: str, buffer: Any, filename: str):
        """Initializes the VAE model object."""
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
        self.vae = VAE(self.args, self.state_shape, self.action_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(),
                                        lr=10 * self.args.lr_ebm,
                                        betas=[0.9, 0.999],
                                        weight_decay=self.args.weight_decay)
        if self.args.image_augmentation == True:
            self.augmentation_transform = RandomCrop(self.imsize, 4, device=device)

    def train(self, batch: Any):
        """Trains the VAE model."""
        stats = []
        self.step += 1
        states = torch.Tensor(batch['observations']).to(device)
        states = states.view(states.shape[0], self.n_channels,
                             self.imsize, self.imsize)
        if self.args.image_augmentation == True:
            states = self.augmentation_transform(states)
        recon_x, mu, logvar = self.vae(states)
        bce = F.binary_cross_entropy(recon_x, states).mean()
        kl = -0.5*torch.mean(1+logvar - mu.pow(2) - logvar.exp())
        loss = bce + 0.005*kl
        print(torch.max(mu), torch.min(mu))
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        stats = {'loss': loss.item()}
        return stats

    def sample_vae(self, init_samples: list = [], sgld_steps: int = None):
        """Samples a latent from VAE."""
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

    def save(self):
        """Saves the model."""
        torch.save(self.vae.state_dict(), self.filename+'model.pt')

    def load(self):
        """Loads the model."""
        self.vae.load_state_dict(torch.load(self.filename))
