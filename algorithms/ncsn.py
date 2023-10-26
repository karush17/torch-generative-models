"""Implements the score net learner."""

from typing import Any

import abc
import torch
import torch.autograd as ag

from models.score_small import ScoreConvNet
from utils.data_utils import RandomCrop
from algorithms.base import BaseTrainer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class NCSNTrainer(BaseTrainer, metaclass=abc.ABCMeta):
    """Implements the noise contrative score net trainer."""
    def __init__(self, args: Any, env: str, buffer: Any, filename: str):
        """Initializes the trainer object."""
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
                                           self.imsize,
                                           self.imsize).uniform_(0, 1)
        self.buffer_size = len(self.reservoir)
        self.score = ScoreConvNet(self.args, self.state_shape,
                                  self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.score.parameters(),
                                        lr=self.args.lr,
                                        betas=[0.9, 0.999],
                                        weight_decay=self.args.weight_decay)
        if self.args.image_augmentation == True:
            self.augmentation_transform = RandomCrop(self.imsize, 4,
                                                     device=device)

    def train(self, batch: Any):
        """Trains the NCSN model."""
        stats = []
        self.step += 1
        states = torch.Tensor(batch['observations']).to(device)
        states = states.view(states.shape[0], self.n_channels,
                             self.imsize, self.imsize)
        if self.args.image_augmentation == True:
            states = self.augmentation_transform(states)
        self.score(states)
        if self.args.score_loss=='ssm':
            dat = states + torch.randn_like(states)*self.args.sgld_std
            loss = self.ssm_loss(dat.detach(), n_particles=1)
        else:
            loss = self.dsm_loss(states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        stats = {'loss': loss.item()}
        return stats

    def ssm_loss(self, dats: torch.Tensor, n_particles: int = 1):
        """Implements the SSM loss."""
        samples = dats.unsqueeze(0).expand(n_particles,
                                           *dats.shape).contiguous().view(-1,
                                                                          *dats.shape[1:])
        samples.requires_grad_(True)
        vectors = torch.randn_like(samples)
        grad1 = self.score(samples)
        gradv = torch.sum(grad1 * vectors)
        grad2 = ag.grad(gradv, samples, create_graph=True)[0]
        grad1 = grad1.view(samples.shape[0], -1)
        loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
        loss2 = torch.sum((vectors * grad2).view(samples.shape[0], -1), dim=-1)
        loss1 = loss1.view(n_particles, -1).mean(dim=0)
        loss2 = loss2.view(n_particles, -1).mean(dim=0)
        loss = loss1 + loss2
        return loss.mean()

    def dsm_loss(self, dats: torch.Tensor):
        """Implements the DSM loss."""
        samples = dats + torch.randn_like(dats)*self.args.noise_std
        target = -1/(self.args.noise_std**2) * (samples - dats)
        scores = self.score(samples)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1/2. * ((scores - target)**2).sum(dim=-1).mean(dim=0)
        return loss

    def sample_persistent(self):
        """Persistent MCMC sampling."""
        random_samples = torch.FloatTensor(self.batch_size,
                                           self.n_channels,
                                           self.imsize,
                                           self.imsize).uniform_(0, 1).to(device)
        if len(self.reservoir) == 0:
            return random_samples
        inds = torch.randint(0, self.buffer_size, (self.batch_size,))
        buffer_samples = self.reservoir[inds].to(device)
        rands = (torch.rand(self.batch_size) <\
                  self.args.reinit_freq).float()[:, None, None, None].to(device)
        samples = rands*random_samples + (1-rands)*buffer_samples
        return samples, inds

    def sample(self, init_samples: list = [], sgld_steps: int = None):
        """Samples from the NCSN model."""
        self.score.eval()
        sgld_lr = 0.00002
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
            grads = self.score(x_k)
            x_k.data += sgld_lr*grads + 0.01*self.args.sgld_std*torch.randn_like(x_k)
            x_k = torch.clamp(x_k, 0, 1)
        self.score.train()
        final_samples = x_k.detach()
        if len(self.reservoir) > 0 and sgld_steps == None:
            self.reservoir[inds] = final_samples.cpu()
        return final_samples

    def save(self):
        """Saves the model."""
        torch.save(self.score.state_dict(), self.filename+'model.pt')

    def load(self):
        """Loads the model."""
        self.score.load_state_dict(torch.load(self.filename))
