import gym
import gym_pepper
import torch as th
import torch.nn as nn
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.pepper_preprocessing import PepperPreprocessing

env = PepperPreprocessing(
    TimeLimit(gym.make("PepperReachCam-v0", gui=False), max_episode_steps=100)
)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim * 3 + 18)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        # Top camera
        self.top_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.bottom_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample()[None]).float()

            top_cam_obs = self._get_top_camera_obs(obs)
            bottom_cam_obs = self._get_bottom_camera_obs(obs)
            depth_cam_obs = self._get_depth_camera_obs(obs)

            top_n_flatten = self.top_cnn(top_cam_obs).shape[1]
            bottom_n_flatten = self.bottom_cnn(bottom_cam_obs).shape[1]
            depth_n_flatten = self.depth_cnn(depth_cam_obs).shape[1]

        self.top_linear = nn.Sequential(
            nn.Linear(top_n_flatten, features_dim), nn.ReLU()
        )
        self.bottom_linear = nn.Sequential(
            nn.Linear(bottom_n_flatten, features_dim), nn.ReLU()
        )
        self.depth_linear = nn.Sequential(
            nn.Linear(depth_n_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        top = self.top_linear(
            self.top_cnn(self._get_top_camera_obs(observations))
        )
        bottom = self.bottom_linear(
            self.bottom_cnn(
                self._get_bottom_camera_obs(observations)
            )
        )
        depth = self.depth_linear(
            self.depth_cnn(self._get_depth_camera_obs(observations))
        )
        joints = self._get_joints_state_obs(observations)

        cat = th.cat((top, bottom, depth, joints), dim=1)

        return cat

    def _get_top_camera_obs(self, obs):
        return obs[:,:3,:,:]

    def _get_bottom_camera_obs(self, obs):
        return obs[:,3:6,:,:]

    def _get_depth_camera_obs(self, obs):
        return obs[:,6,:,:].unsqueeze(dim=1)

    def _get_joints_state_obs(self, obs):
        return obs[:,7,:18,0]


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
    buffer_size=10000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    ent_coef="auto",
    policy_kwargs=policy_kwargs,
    train_freq=1,
    tensorboard_log="./data/1_tensorboard/",
)


model.learn(10000)

model.save("./data/1")
