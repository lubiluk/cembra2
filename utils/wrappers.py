# Copied from https://github.com/araffin/rl-baselines-zoo
import gym
from gym import spaces
from gym.wrappers import TimeLimit
import numpy as np
import torch
import torchvision.transforms as transforms


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low,
                                               high=high,
                                               dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


class TorchifyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TorchifyWrapper, self).__init__(env)

        self.img_size = (1, 120, 160)

        obs_spaces = dict(
            camera=gym.spaces.Box(
                0.0,
                1.0,
                shape=self.img_size,
                dtype=np.float32,
            ),
            camera_pose=self.observation_space.spaces["camera_pose"],
            joints_state=self.observation_space.spaces["joints_state"],
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    def observation(self, obs):
        """what happens to each observation"""

        # Convert image to grayscale
        img = obs["camera"]

        obs["camera"] = self.transform(img)

        return obs


class VisionWrapper(gym.ObservationWrapper):
    def __init__(self, env, net_class, model_file):
        super(VisionWrapper, self).__init__(env)

        self.net = net_class()
        self.net.load_state_dict(
            torch.load(model_file, map_location=torch.device('cpu')))
        self.net.eval()

        with torch.no_grad():
            cam_space = env.observation_space.spaces["camera"]
            obs = torch.as_tensor(cam_space.sample()[None]).float()
            cam_dim = self.net(obs).shape[1]

        cam_pose_dim = self.observation_space.spaces["camera_pose"].shape[0]
        joints_state_dim = self.observation_space.spaces["joints_state"].shape[
            0]

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            shape=(cam_dim + cam_pose_dim +
                                                   joints_state_dim, 1),
                                            dtype="float32")

    def observation(self, obs):
        """what happens to each observation"""

        # Convert image to grayscale
        img = obs["camera"]
        img_feat = self.net(img.unsqueeze(dim=0)).squeeze(dim=0)
        cam_pose = torch.as_tensor(obs["camera_pose"])
        joints_state = torch.as_tensor(obs["joints_state"])

        print(img_feat)

        return torch.cat((joints_state, cam_pose, img_feat))


class BaselinifyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(BaselinifyWrapper, self).__init__(env)

        self.img_size = (1, 120, 160)
        self.observation_space = camera = gym.spaces.Box(
            0.0,
            1.0,
            shape=(2, 120, 160),
            dtype=np.float32,
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    def observation(self, obs):
        """what happens to each observation"""

        # Convert image to grayscale
        img = obs["camera"]
        img = self.transform(img)

        joints_state = obs['joints_state']
        cam_pose = obs['camera_pose']
        zeros = np.zeros(img.shape[1:])
        zeros[:joints_state.shape[0], 0] = joints_state
        zeros[joints_state.shape[0]:joints_state.shape[0] + cam_pose.shape[0],
              0] = cam_pose

        return np.concatenate([img, np.expand_dims(zeros, axis=0)])