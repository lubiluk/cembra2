import torch
import numpy as np
from .utils import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_space, act_dim, env=None, size=100000, device=None):
        self.device = device

        camera_dim = obs_space.spaces["camera"].shape
        camera_pose_dim = obs_space.spaces["camera_pose"].shape
        joints_state_dim = obs_space.spaces["joints_state"].shape

        self.obs_camera_buf = torch.zeros(combined_shape(size, camera_dim),
                                       dtype=torch.float32,
                                       device=device)
        self.obs_camera_pose_buf = torch.zeros(combined_shape(size, camera_pose_dim),
                                       dtype=torch.float32,
                                       device=device)
        self.obs_joints_state_buf = torch.zeros(combined_shape(size, joints_state_dim),
                                       dtype=torch.float32,
                                       device=device)
        self.obs2_camera_buf = torch.zeros(combined_shape(size, camera_dim),
                                        dtype=torch.float32,
                                        device=device)
        self.obs2_camera_pose_buf = torch.zeros(combined_shape(size, camera_pose_dim),
                                       dtype=torch.float32,
                                       device=device)
        self.obs2_joints_state_buf = torch.zeros(combined_shape(size, joints_state_dim),
                                        dtype=torch.float32,
                                        device=device)
        self.act_buf = torch.zeros(combined_shape(size, act_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, info):
        obs_camera = obs["camera"]
        obs_camera_pose = obs["camera_pose"]
        obs_joints_state = obs["joints_state"]

        next_obs_camera = next_obs["camera"]
        next_obs_camera_pose = next_obs["camera_pose"]
        next_obs_joints_state = next_obs["joints_state"]

        self.obs_camera_buf[self.ptr] = torch.as_tensor(obs_camera,
                                                     dtype=torch.float32)
        self.obs_camera_pose_buf[self.ptr] = torch.as_tensor(obs_camera_pose,
                                                     dtype=torch.float32)
        self.obs_joints_state_buf[self.ptr] = torch.as_tensor(obs_joints_state,
                                                     dtype=torch.float32)
        self.obs2_camera_buf[self.ptr] = torch.as_tensor(next_obs_camera,
                                                      dtype=torch.float32)
        self.obs2_camera_pose_buf[self.ptr] = torch.as_tensor(next_obs_camera_pose,
                                                      dtype=torch.float32)
        self.obs2_joints_state_buf[self.ptr] = torch.as_tensor(next_obs_joints_state,
                                                      dtype=torch.float32)
        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)
        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = {
            "camera": self.obs_camera_buf[idxs],
            "camera_pose": self.obs_camera_pose_buf[idxs],
            "joints_state": self.obs_joints_state_buf[idxs]
        }
        obs2 = {
            "camera": self.obs2_camera_buf[idxs],
            "camera_pose": self.obs2_camera_pose_buf[idxs],
            "joints_state": self.obs2_joints_state_buf[idxs]
        }

        return dict(obs=obs,
                    obs2=obs2,
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

    def start_episode(self):
        pass

    def end_episode(self):
        pass