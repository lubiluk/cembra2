import numpy as np
import torch.nn as nn
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, ) + shape


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def cnn(sizes, activation, output_size, cam_space):
    layers = []

    for j in range(len(sizes)):
        layers += [
            nn.Conv2d(sizes[j][0],
                      sizes[j][1],
                      kernel_size=sizes[j][2],
                      stride=sizes[j][3],
                      padding=sizes[j][4]),
            activation()
        ]

    conv = nn.Sequential(*(layers + [nn.Flatten()]))

    # Compute shape by doing one forward pass
    with torch.no_grad():
        obs = torch.as_tensor(cam_space.sample()[None]).float()
        bottom_n_flatten = conv(obs).shape[1]

    lin = nn.Sequential(nn.Linear(bottom_n_flatten, output_size), nn.ReLU())

    return (conv, lin)

def output_n(conv, cam_space):
    with torch.no_grad():
        obs = torch.as_tensor(cam_space.sample()[None]).float()
        return conv(obs).shape[1]