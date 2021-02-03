import numpy as np
import torch
import torch.nn as nn


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


class MLPActor(nn.Module):
    def __init__(self,
                 obs_space,
                 act_dim,
                 hidden_sizes,
                 activation,
                 act_limit,
                 device=None):
        super(MLPActor, self).__init__()

        obs_dim = obs_space.spaces["observation"].shape[0]
        dgoal_dim = obs_space.spaces["desired_goal"].shape[0]

        pi_sizes = [obs_dim + dgoal_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

        if device:
            self.pi.to(device)

    def forward(self, obs):
        obs_lin = torch.stack([o["observation"] for o in obs])
        dgoal = torch.stack([o["desired_goal"] for o in obs])
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(torch.cat((obs_lin, dgoal), dim=-1))


class MLPQFunction(nn.Module):
    def __init__(self,
                 obs_space,
                 act_dim,
                 hidden_sizes,
                 activation,
                 device=None):
        super(MLPQFunction, self).__init__()

        obs_dim = obs_space.spaces["observation"].shape[0]
        dgoal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.q = mlp([obs_dim + dgoal_dim + act_dim] + list(hidden_sizes) + [1],
                     activation)

        if device:
            self.q.to(device)

    def forward(self, obs, act):
        obs_lin = torch.stack([o["observation"] for o in obs])
        dgoal = torch.stack([o["desired_goal"] for o in obs])

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
        self.pi = MLPActor(observation_space,
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

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a = self.pi(obs.unsqueeze(dim=0))
            return a.squeeze(dim=0).cpu().numpy()
            # return self.pi(obs).numpy()
