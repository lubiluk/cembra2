import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gym.spaces as spaces


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


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_space, act_dim, hidden_sizes, activation, act_limit, extractor, e_kwargs):
        super(SquashedGaussianMLPActor, self).__init__()

        if extractor:
            self.ext = extractor(obs_space, **e_kwargs)
            self.net = mlp([self.ext.feature_space.shape[0]] + list(hidden_sizes), activation, activation)
        else:
            self.ext = None
            self.net = mlp([obs_space.shape[0]] + list(hidden_sizes), activation, activation)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        if self.ext:
            net_out = self.net(self.ext(obs))
        else:
            net_out = self.net(obs)
        
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (
                2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                    axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_dim, hidden_sizes, activation, extractor, e_kwargs):
        super(MLPQFunction, self).__init__()
        if extractor:
            self.ext = extractor(obs_space, **e_kwargs)
            self.q = mlp([self.ext.feature_space.shape[0] + act_dim] + list(hidden_sizes) + [1],
                        activation)
        else:
            self.ext = None
            self.q = mlp([obs_space.shape[0] + act_dim] + list(hidden_sizes) + [1], activation)
            
    def forward(self, obs, act):
        if self.ext:
            q = self.q(torch.cat([self.ext(obs), act], dim=-1))
        else:
            q = self.q(torch.cat([obs, act], dim=-1))

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 extractor=None,
                 e_kwargs=dict()):
        super(MLPActorCritic, self).__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(observation_space, act_dim, hidden_sizes,
                                           activation, act_limit, extractor=extractor, e_kwargs=e_kwargs)
        self.q1 = MLPQFunction(observation_space, act_dim, hidden_sizes, activation, extractor=extractor, e_kwargs=e_kwargs)
        self.q2 = MLPQFunction(observation_space, act_dim, hidden_sizes, activation, extractor=extractor, e_kwargs=e_kwargs)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs.unsqueeze(dim=0), deterministic, False)
            return a.squeeze(dim=0).numpy()


def cnn(sizes, activation):
    return nn.Sequential(nn.Linear(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ), features_dim), nn.ReLU())

class FeatureExtractor(nn.Module):
    def __init__(self,
                 observation_space,
                 cnn_sizes=(256, 256),
                 activation=nn.ReLU,
                 features_dim=256):
        super(FeatureExtractor, self).__init__()

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.bottom_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            obs = torch.as_tensor(observation_space.sample()[None]).float()
            bottom_cam_obs = self._get_bottom_camera_obs(obs)
            bottom_n_flatten = self.bottom_cnn(bottom_cam_obs).shape[1]

        self.bottom_linear = nn.Sequential(
            nn.Linear(bottom_n_flatten, features_dim), nn.ReLU())

        self.feature_space = spaces.Box(-np.inf, np.inf, (features_dim + 18, ),
                               np.float32)

    def forward(self, observations):
        bottom = self.bottom_linear(
            self.bottom_cnn(
                self._get_bottom_camera_obs(observations)
            )
        )
        joints = self._get_joints_state_obs(observations)

        cat = torch.cat((bottom, joints), dim=1)

        return cat

    def _get_bottom_camera_obs(self, obs):
        return obs[:,:3,:,:]

    def _get_joints_state_obs(self, obs):
        return obs[:,3,:18,0]

