import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ..common.utils import mlp

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self,
                 obs_space,
                 act_dim,
                 hidden_sizes,
                 activation,
                 act_limit,
                 device=None):
        super(SquashedGaussianMLPActor, self).__init__()
        self.device = device

        obs_dim = obs_space.spaces["observation"].shape[0]
        dgoal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.net = mlp([obs_dim + dgoal_dim] + list(hidden_sizes), activation,
                       activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

        if device:
            self.net.to(device)
            self.mu_layer.to(device)
            self.log_std_layer.to(device)

    def forward(self, obs, deterministic=False, with_logprob=True):
        obs_lin = torch.as_tensor(obs["observation"],
                                  dtype=torch.float32,
                                  device=self.device)
        dgoal = torch.as_tensor(obs["desired_goal"],
                                dtype=torch.float32,
                                device=self.device)

        net_out = self.net(torch.cat((obs_lin, dgoal), dim=-1))
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
    def __init__(self,
                 obs_space,
                 act_dim,
                 hidden_sizes,
                 activation,
                 device=None):
        super(MLPQFunction, self).__init__()
        self.device = device

        obs_dim = obs_space.spaces["observation"].shape[0]
        dgoal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.q = mlp([obs_dim + dgoal_dim + act_dim] + list(hidden_sizes) +
                     [1], activation)

        if device:
            self.q.to(device)

    def forward(self, obs, act):
        obs_lin = torch.as_tensor(obs["observation"],
                                  dtype=torch.float32,
                                  device=self.device)
        dgoal = torch.as_tensor(obs["desired_goal"],
                                dtype=torch.float32,
                                device=self.device)

        q = self.q(torch.cat([obs_lin, dgoal, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 device=None):
        super(MLPActorCritic, self).__init__()

        self.device = device

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(observation_space,
                                           act_dim,
                                           hidden_sizes,
                                           activation,
                                           act_limit,
                                           device=device)
        self.q1 = MLPQFunction(observation_space,
                               act_dim,
                               hidden_sizes,
                               activation,
                               device=device)
        self.q2 = MLPQFunction(observation_space,
                               act_dim,
                               hidden_sizes,
                               activation,
                               device=device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs["observation"] = [obs["observation"]]
            obs["desired_goal"] = [obs["desired_goal"]]
            obs["achieved_goal"] = [obs["achieved_goal"]]

            a, _ = self.pi(obs, deterministic, False)
            return a.squeeze(dim=0).cpu().numpy()
