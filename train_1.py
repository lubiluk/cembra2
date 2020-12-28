import gym
import gym_pepper
import torch as th
import torch.nn as nn
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.pepper_preprocessing import PepperPreprocessing

env = PepperPreprocessing(
    TimeLimit(gym.make("PepperReachCam-v0"), max_episode_steps=100)
)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    activation_fn=th.nn.ReLU,
    net_arch=[256, 256, 256],
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=1000000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    ent_coef="auto",
    policy_kwargs=policy_kwargs,
    train_freq=1,
    tensorboard_log="./data/1_tensorboard/",
)


model.learn(1000)

model.save("./data/1")
