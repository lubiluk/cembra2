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


class MLPActor(nn.Module):
    def __init__(self,
                 obs_space,
                 act_dim,
                 conv_sizes,
                 feature_dim,
                 hidden_sizes,
                 activation,
                 act_limit,
                 device=None):
        super(MLPActor, self).__init__()

        cam_space = obs_space.spaces["observation"]["camera_bottom"]
        joints_dim = obs_space.spaces["observation"]["joints_state"].shape[0]
        dgoal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.cnn, self.feat = cnn(conv_sizes, activation, feature_dim,
                                  cam_space)
        pi_sizes = [feature_dim + joints_dim + dgoal_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        obs_img = torch.stack([o["observation"]["camera_bottom"] for o in obs])
        obs_img_nom = obs_img / 255 - 0.5 # normalize
        obs_lin = torch.stack([o["observation"]["joints_state"] for o in obs])
        dgoal = torch.stack([o["desired_goal"] for o in obs])

        feat = self.feat(self.cnn(obs_img_nom))

        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(torch.cat((feat, obs_lin, dgoal), dim=-1))


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
        
        cam_space = obs_space.spaces["observation"]["camera_bottom"]
        joints_dim = obs_space.spaces["observation"]["joints_state"].shape[0]
        dgoal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.cnn, self.feat = cnn(conv_sizes, activation, feature_dim,
                                  cam_space)
        self.q = mlp([feature_dim + joints_dim + dgoal_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        obs_img = torch.stack([o["observation"]["camera_bottom"] for o in obs])
        obs_img_nom = obs_img / 255 - 0.5 # normalize
        obs_lin = torch.stack([o["observation"]["joints_state"] for o in obs])
        dgoal = torch.stack([o["desired_goal"] for o in obs])

        feat = self.feat(self.cnn(obs_img_nom))

        q = self.q(torch.cat((feat, obs_lin, dgoal, act), dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


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
        self.pi = MLPActor(observation_space, act_dim, conv_sizes, feature_dim,
                           hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(observation_space, act_dim, conv_sizes,
                               feature_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, act_dim, conv_sizes,
                               feature_dim, hidden_sizes, activation)

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a = self.pi(obs.unsqueeze(dim=0))
            return a.squeeze(dim=0).cpu().numpy()
            # return self.pi(obs).numpy()
