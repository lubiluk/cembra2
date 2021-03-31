import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 linear_dim: int,
                 features_dim: int = 256,
                 n_channels: int = 1):
        super(CustomCNN, self).__init__(observation_space,
                                        features_dim + linear_dim)

        self.linear_dim = linear_dim
        self.n_channels = n_channels

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 4, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample()[None]).float()
            cam_obs = self._get_camera_obs(obs)
            n_flatten = self.cnn(cam_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cam = self.linear(self.cnn(self._get_camera_obs(observations)))
        lin = self._get_linear_obs(observations)

        cat = th.cat((cam, lin), dim=1)

        return cat

    def _get_camera_obs(self, obs):
        return obs[:, :self.n_channels, :, :]

    def _get_linear_obs(self, obs):
        return obs[:, self.n_channels, :self.linear_dim, 0]
