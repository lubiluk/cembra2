import torch
import numpy as np
from gym import spaces
from gym.spaces.box import Box
from gym.core import Wrapper


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, use_gpu=True):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)

        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            if use_gpu:
                self.device = torch.device("cuda")
                print('\nUsing GPU framebuffer\n')

        n_channels, height, width = env.observation_space.spaces[
            "camera_bottom"].shape
        obs_shape = [n_channels * n_frames, height, width]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = torch.zeros(obs_shape, dtype=torch.float32, device=self.device)

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
        self.framebuffer = torch.zeros_like(self.framebuffer,
                                            dtype=torch.float32,
                                            device=self.device)
        obs = self.env.reset()
        new_img = obs["camera_bottom"]
        self.update_buffer(new_img)
        obs["camera_bottom"] = self.framebuffer
        return obs

    def step(self, action):
        """plays for 1 step, returns frame buffer"""
        obs, reward, done, info = self.env.step(action)
        new_img = torch.as_tensor(obs["camera_bottom"],
                                  dtype=torch.float32,
                                  device=self.device)
        self.update_buffer(new_img)
        obs["camera_bottom"] = self.framebuffer
        return obs, reward, done, info

    def update_buffer(self, img):
        offset = self.env.observation_space.spaces["camera_bottom"].shape[0]
        axis = 0
        cropped_framebuffer = self.framebuffer[:-offset]
        self.framebuffer = torch.cat([img, cropped_framebuffer], dim=axis)
