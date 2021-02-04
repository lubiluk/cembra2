# Copied from https://github.com/araffin/rl-baselines-zoo
import gym
from gym.wrappers import TimeLimit
import numpy as np
from PIL import Image


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


class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        super(GrayscaleWrapper, self).__init__(env)

        self.img_size = (120, 160, 1)

        obs_spaces = dict(
            camera_bottom=gym.spaces.Box(
                0.0,
                1.0,
                shape=self.img_size,
                dtype=np.float32,
            ),
            joints_state=self.observation_space.spaces["joints_state"],
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def observation(self, obs):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        img = obs["camera_bottom"]
        img = Image.fromarray(img)
        img = img.convert('L')
        img = np.expand_dims(np.array(img, dtype=np.float32), axis=-1)
        img = img / 255

        obs["camera_bottom"] = img

        return obs


# in torch imgs have shape [c, h, w] instead of common [h, w, c]
class AntiTorchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(AntiTorchWrapper, self).__init__(env)

        self.img_size = [
            env.observation_space.spaces["camera_bottom"].shape[i]
            for i in [2, 0, 1]
        ]

        obs_spaces = dict(
            camera_bottom=gym.spaces.Box(
                0.0,
                1.0,
                shape=self.img_size,
                dtype=np.float32,
            ),
            joints_state=self.observation_space.spaces["joints_state"],
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def observation(self, obs):
        """what happens to each observation"""
        img = obs["camera_bottom"]
        img = img.transpose(2, 0, 1)
        obs["camera_bottom"] = img

        return obs