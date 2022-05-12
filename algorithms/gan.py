import abc
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.gan import Generator, Discriminator
from utils.data_utils import grad_norm, plot, RandomCrop
from algorithms.base import BaseTrainer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class GANTrainer(BaseTrainer, metaclass=abc.ABCMeta):
    def __init__(self, args, env, buffer, filename):
        self.args = args
        self.batch_size = self.args.batch_size
        self.filename = filename
        self.env = env
        self.buffer = buffer
        self.step = 0
        self.state_shape = self.env.reset()['image'].shape
        self.n_channels = self.state_shape[0]
        self.imsize = self.state_shape[1]
        self.im_len = self.n_channels*self.imsize*self.imsize
        self.h_dim = 100
        self.action_dim = self.env.action_space.low.size
        self.gen = Generator(self.args, self.state_shape, self.action_dim).to(device)
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.args.lr, \
                            betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        self.disc = Discriminator(self.args, self.state_shape, self.action_dim).to(device)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.args.lr, \
                            betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        if self.args.image_augmentation == True:
            self.augmentation_transform = RandomCrop(self.imsize, 4, device=device)
        self.adv_loss = nn.BCELoss()

    def train(self, batch):
        stats = []
        self.step += 1
        states = torch.Tensor(batch['observations']).to(device)
        states = states.view(states.shape[0], self.n_channels, self.imsize, self.imsize)
        if self.args.image_augmentation == True:
            states = self.augmentation_transform(states)
        self.decay_lr()
        valid = Variable(torch.Tensor(states.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(states.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        real_imgs = states

        self.gen_optimizer.zero_grad()
        z = Variable(torch.Tensor(np.random.normal(0, 1, (states.shape[0], self.h_dim)))).to(device)
        gen_imgs = self.gen(z)
        g_loss = self.adv_loss(self.disc(gen_imgs), valid)
        g_loss.backward()
        self.gen_optimizer.step()

        self.disc_optimizer.zero_grad()
        real_loss = self.adv_loss(self.disc(real_imgs), valid)
        fake_loss = self.adv_loss(self.disc(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.disc_optimizer.step()
        stats = {'gen_loss': g_loss.item(), 'disc_loss': d_loss.item()}
        return stats
    
    def sample_persistent(self):
        random_samples = torch.FloatTensor(self.batch_size, self.n_channels, self.imsize, self.imsize).uniform_(0, 1).to(device)
        if len(self.reservoir) == 0:
            return random_samples
        buffer_size = len(self.reservoir)
        inds = torch.randint(0, self.buffer_size, (self.batch_size,))
        buffer_samples = self.reservoir[inds].to(device)
        rands = (torch.rand(self.batch_size) < self.args.reinit_freq).float()[:, None, None, None].to(device)
        samples = rands*random_samples + (1-rands)*buffer_samples
        return samples, inds
    
    def sample(self, init_samples=[]):
        self.gen.eval()
        init_samples = torch.Tensor(init_samples).to(device)
        samples = self.gen(init_samples)
        self.gen.train()
        return samples.detach()
    
    def decay_lr(self):
        if self.step in self.args.decay_epochs:
            for param_group in self.gen_optimizer.param_groups:
                new_lr = param_group['lr'] * self.args.decay_rate
                param_group['lr'] = new_lr

    def save(self):
        torch.save(self.gen.state_dict(), self.filename+'model.pt')
    
    def load(self):
        self.gen.load_state_dict(torch.load(self.filename))
