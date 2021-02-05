import numpy as np
from gym import spaces
from gym.spaces.box import Box
from gym.core import Wrapper


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)

        n_channels, height, width = env.observation_space.spaces["camera_bottom"].shape
        obs_shape = [n_channels * n_frames, height, width]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

        obs_spaces = dict(
            camera_bottom=spaces.Box(
                0.0,
                1.0,
                shape=(n_channels * n_frames, height, width),
                dtype=np.float32,
            ),
            joints_state=env.observation_space.spaces["joints_state"],
        )

        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self):
        """resets, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        obs = self.env.reset()
        new_img = obs["camera_bottom"]
        self.update_buffer(new_img)
        obs["camera_bottom"] = self.framebuffer
        return obs

    def step(self, action):
        """plays for 1 step, returns frame buffer"""
        obs, reward, done, info = self.env.step(action)
        new_img = obs["camera_bottom"]
        self.update_buffer(new_img)
        obs["camera_bottom"] = self.framebuffer
        return obs, reward, done, info

    def update_buffer(self, img):
        offset = self.env.observation_space.spaces["camera_bottom"].shape[0]
        axis = 0
        cropped_framebuffer = self.framebuffer[:-offset]
        self.framebuffer = np.concatenate(
            [img, cropped_framebuffer], axis=axis)
