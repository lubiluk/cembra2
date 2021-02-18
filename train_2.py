import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core_cam
from algos.common import replay_buffer_cam
from gym.wrappers.time_limit import TimeLimit
from utils.wrappers import TorchifyWrapper


env = TorchifyWrapper(
    TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
              max_episode_steps=100))

ac_kwargs = dict(hidden_sizes=[128, 128],
                 activation=nn.ReLU,
                 conv_sizes=[[1, 8, 2, 1, 0], [8, 16, 2, 1, 0]],
                 feature_dim=32)
rb_kwargs = dict(size=30000)

logger_kwargs = dict(output_dir='data/2', exp_name='2')

model = SAC(env=env,
            actor_critic=core_cam.MLPActorCritic,
            ac_kwargs=ac_kwargs,
            replay_buffer=replay_buffer_cam.ReplayBuffer,
            rb_kwargs=rb_kwargs,
            max_ep_len=100,
            batch_size=256,
            gamma=0.95,
            lr=0.001,
            update_after=256,
            update_every=256,
            logger_kwargs=logger_kwargs)

model.train(steps_per_epoch=1000, epochs=100)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/2')

run_policy(env, get_action)