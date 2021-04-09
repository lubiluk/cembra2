import torch
import numpy as np
from .utils import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for HER agents.
    """
    def __init__(self,
                 obs_space,
                 act_dim,
                 env,
                 size=100000,
                 n_sampled_goal=1,
                 goal_selection_strategy='final',
                 device=None):
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy
        self.device = device

        obs_dim = obs_space.spaces["observation"].shape[0]
        goal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.obs_buf = torch.zeros(combined_shape(size, obs_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.obs_dgoal_buf = torch.zeros(combined_shape(size, goal_dim),
                                         dtype=torch.float32,
                                         device=device)
        self.obs_agoal_buf = torch.zeros(combined_shape(size, goal_dim),
                                         dtype=torch.float32,
                                         device=device)

        self.act_buf = torch.zeros(combined_shape(size, act_dim),
                                   dtype=torch.float32,
                                   device=device)

        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim),
                                    dtype=torch.float32,
                                    device=device)
        self.obs2_dgoal_buf = torch.zeros(combined_shape(size, goal_dim),
                                          dtype=torch.float32,
                                          device=device)
        self.obs2_agoal_buf = torch.zeros(combined_shape(size, goal_dim),
                                          dtype=torch.float32,
                                          device=device)

        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.info_buf = np.empty((size, 1), dtype=object)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.ep_start_ptr = self.ptr

    def store(self, obs, act, rew, next_obs, done, info):
        obs_lin = obs["observation"]
        obs_dgoal = obs["desired_goal"]
        obs_agoal = obs["achieved_goal"]

        next_obs_lin = next_obs["observation"]
        next_obs_dgoal = next_obs["desired_goal"]
        next_obs_agoal = next_obs["achieved_goal"]

        self.obs_buf[self.ptr] = torch.as_tensor(obs_lin, dtype=torch.float32)
        self.obs_dgoal_buf[self.ptr] = torch.as_tensor(obs_dgoal,
                                                       dtype=torch.float32)
        self.obs_agoal_buf[self.ptr] = torch.as_tensor(obs_agoal,
                                                       dtype=torch.float32)

        self.obs2_buf[self.ptr] = torch.as_tensor(next_obs_lin,
                                                  dtype=torch.float32)
        self.obs2_dgoal_buf[self.ptr] = torch.as_tensor(next_obs_dgoal,
                                                        dtype=torch.float32)
        self.obs2_agoal_buf[self.ptr] = torch.as_tensor(next_obs_agoal,
                                                        dtype=torch.float32)

        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)

        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.info_buf[self.ptr] = info

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == self.ep_start_ptr:
            raise "Episode longer than buffer size"

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return self._get_batch(idxs)

    def start_episode(self):
        self.ep_start_ptr = self.ptr

    def end_episode(self):
        self._synthesize_experience()

    def _get_current_episode(self):
        if self.ep_start_ptr == self.ptr:
            return [self.ptr]

        if self.ep_start_ptr <= self.ptr:
            idxs = np.arange(self.ep_start_ptr, self.ptr)
        else:
            idxs = np.concatenate(
                [np.arange(self.ep_start_ptr, self.size),
                 np.arange(self.ptr)])

        return self._get_batch(idxs)

    def _get_batch(self, idxs):
        obs = {
            "observation": self.obs_buf[idxs],
            "desired_goal": self.obs_dgoal_buf[idxs],
            "achieved_goal": self.obs_agoal_buf[idxs]
        }
        obs2 = {
            "observation": self.obs2_buf[idxs],
            "desired_goal": self.obs2_dgoal_buf[idxs],
            "achieved_goal": self.obs2_agoal_buf[idxs]
        }

        return dict(obs=obs,
                    obs2=obs2,
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs],
                    info=self.info_buf[idxs])

    def _synthesize_experience(self):
        ep = self._get_current_episode()
        ep_len = len(ep['rew'])

        for idx in range(ep_len):
            obs = {
                "observation": ep["obs"]["observation"][idx],
                "desired_goal": ep["obs"]["desired_goal"][idx],
                "achieved_goal": ep["obs"]["achieved_goal"][idx]
            }
            act = ep['act'][idx]
            obs2 = {
                "observation": ep["obs2"]["observation"][idx],
                "desired_goal": ep["obs2"]["desired_goal"][idx],
                "achieved_goal": ep["obs2"]["achieved_goal"][idx]
            }
            info = ep['info'][idx]
            np_agoal = obs2['achieved_goal'].cpu().numpy()

            for _ in range(self.n_sampled_goal):
                if self.selection_strategy == 'final':
                    sel_idx = -1
                elif self.selection_strategy == 'future':
                    # We cannot sample a goal from the future in the last step of an episode
                    if idx == ep_len - 1:
                        break
                    sel_idx = np.random.choice(np.arange(idx + 1, ep_len))
                elif self.selection_strategy == 'episode':
                    sel_idx = np.random.choice(np.arange(ep_len))
                else:
                    raise ValueError(
                        "Unsupported selection_strategy: {}".format(
                            self.selection_strategy))

                sel_agoal = ep["obs2"]["achieved_goal"][sel_idx]
                info = ep['info'][sel_idx]
                done = ep['done'][sel_idx]
                np_sel_agoal = sel_agoal.cpu().numpy()

                rew = self.env.compute_reward(np_agoal, np_sel_agoal, info)

                obs["desired_goal"] = sel_agoal
                obs2["desired_goal"] = sel_agoal

                self.store(obs, act, rew, obs2, done, info)