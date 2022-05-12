import torch
import torchvision
import torch.nn as nn
import numpy as np
from utils.obs_buffer import ObsDictReplayBuffer

sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

def sqrt(x):
    return int(torch.sqrt(torch.Tensor([x])))

def plot(p, img):
    torchvision.utils.save_image(torch.clamp(img, -1, 1), p, normalize=True)

def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()

def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions

def add_data_to_buffer(data, replay_buffer):
    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))
        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_images(data[j]['observations']),
            next_observations=process_images(
                data[j]['next_observations']),
        )
        replay_buffer.add_path(path)

def load_data_from_npy(variant, expl_env, observation_key,
                       extra_buffer_size=100):
    with open(variant['buffer'], 'rb') as f:
        data = np.load(f, allow_pickle=True)
    num_transitions = get_buffer_size(data)
    buffer_size = num_transitions + extra_buffer_size
    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
    )
    add_data_to_buffer(data, replay_buffer)
    print('Data loaded from npy file', replay_buffer._top)
    return replay_buffer

def load_data_from_npy_chaining(variant, expl_env, observation_key,
                                extra_buffer_size=100):
    with open(variant.buffer, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    buffer_size = get_buffer_size(data)
    buffer_size += extra_buffer_size
    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
    )
    add_data_to_buffer(data, replay_buffer)
    top = replay_buffer._top
    print('Data loaded from npy file', top)
    # replay_buffer._rewards[:top] = 0.0*replay_buffer._rewards[:top]
    # print('Zero-ed the rewards for prior data', top)

    return replay_buffer

def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))
    return output

class RandomCrop:
    """
    Source: # https://github.com/pratogab/batch-transforms
    Applies the :class:`~torchvision.transforms.RandomCrop` transform to
    a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1),
                                  tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2),
                                 dtype=self.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),),
                              device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),),
                              device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:,
                                                                        None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:,
                                                                           None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None],
                 rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded.permute(1, 0, 2, 3)