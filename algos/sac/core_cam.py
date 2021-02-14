import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ..common.utils import mlp, cnn

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self,
                 obs_space,
                 act_dim,
                 conv_sizes,
                 feature_dim,
                 hidden_sizes,
                 activation,
                 act_limit,
                 device=None):
        super(SquashedGaussianMLPActor, self).__init__()
        self.device = device

        cam_space = obs_space.spaces["camera_bottom"]
        joints_dim = obs_space.spaces["joints_state"].shape[0]

        self.cnn, self.feat = cnn(conv_sizes, activation, feature_dim,
                                  cam_space)
        self.net = mlp([feature_dim + joints_dim] + list(hidden_sizes),
                       activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

        if device:
            self.cnn.to(device)
            self.feat.to(device)
            self.net.to(device)
            self.mu_layer.to(device)
            self.log_std_layer.to(device)

    def forward(self, obs, deterministic=False, with_logprob=True):
        out_device = obs["camera_bottom"][0].device
        obs_img = obs["camera_bottom"].to(self.device)
        obs_lin = obs["joints_state"].to(self.device)

        feat = self.feat(self.cnn(obs_img))

        net_out = self.net(torch.cat((feat, obs_lin), dim=-1))
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

        return pi_action.to(out_device), logp_pi


class MLPQFunction(nn.Module):
    def __init__(self,
                 obs_space,
                 act_dim,
                 conv_sizes,
                 feature_dim,
                 hidden_sizes,
                 activation,
                 device=None):
        super(MLPQFunction, self).__init__()

        self.device = device

        cam_space = obs_space.spaces["camera_bottom"]
        joints_dim = obs_space.spaces["joints_state"].shape[0]

        self.cnn, self.feat = cnn(conv_sizes, activation, feature_dim,
                                  cam_space)
        self.q = mlp([feature_dim + joints_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)

        if device:
            self.cnn.to(device)
            self.feat.to(device)
            self.q.to(device)

    def forward(self, obs, act):
        out_device = act.device

        obs_img = obs["camera_bottom"].to(self.device)
        obs_lin = obs["joints_state"].to(self.device)

        feat = self.feat(self.cnn(obs_img))

        act = act.to(self.device)

        q = self.q(torch.cat((feat, obs_lin, act), dim=-1))
        return torch.squeeze(q, -1).to(
            out_device)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 conv_sizes=((3, 32, 8, 4, 0), (32, 64, 4, 2, 0)),
                 feature_dim=256,
                 device=None):
        super(MLPActorCritic, self).__init__()

        self.device = device

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(observation_space,
                                           act_dim,
                                           conv_sizes,
                                           feature_dim,
                                           hidden_sizes,
                                           activation,
                                           act_limit,
                                           device=device)
        self.q1 = MLPQFunction(observation_space,
                               act_dim,
                               conv_sizes,
                               feature_dim,
                               hidden_sizes,
                               activation,
                               device=device)
        self.q2 = MLPQFunction(observation_space,
                               act_dim,
                               conv_sizes,
                               feature_dim,
                               hidden_sizes,
                               activation,
                               device=device)

    def act(self, obs, deterministic=False):
        obs = {
            "camera_bottom": obs["camera_bottom"].unsqueeze(dim=0),
            "joints_state": obs["joints_state"].unsqueeze(dim=0),
        }

        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.squeeze(dim=0).cpu().numpy()
