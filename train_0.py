import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core
from algos.common import replay_buffer
from gym.wrappers.time_limit import TimeLimit

env = TimeLimit(gym.make("PepperReach-v0", gui=False, dense=True),
                max_episode_steps=100)

ac_kwargs = dict(hidden_sizes=[64, 64, 64], activation=nn.ReLU)
rb_kwargs = dict(size=1000000)

logger_kwargs = dict(output_dir='data/0', exp_name='0')

model = SAC(env=env,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=ac_kwargs,
            replay_buffer=replay_buffer.ReplayBuffer,
            rb_kwargs=rb_kwargs,
            max_ep_len=100,
            batch_size=64,
            gamma=0.95,
            lr=0.001,
            update_after=1000,
            update_every=64,
            logger_kwargs=logger_kwargs)

model.train(steps_per_epoch=10000, epochs=100)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/0')

run_policy(env, get_action)
