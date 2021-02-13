# Sanity test on Fetch
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core
from algos.common import replay_buffer

env = gym.make('Pendulum-v0')

ac_kwargs = dict(hidden_sizes=[256, 256], activation=nn.ReLU)
rb_kwargs = dict(size=1000000)

logger_kwargs = dict(output_dir='data/sanity_1', exp_name='sanity_1')

model = SAC(env=env,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=ac_kwargs,
            replay_buffer=replay_buffer.ReplayBuffer,
            rb_kwargs=rb_kwargs,
            max_ep_len=1000,
            batch_size=256,
            gamma=0.99,
            lr=0.0003,
            polyak=0.995,
            update_after=100,
            update_every=1,
            logger_kwargs=logger_kwargs)

model.train(steps_per_epoch=1000, epochs=10)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/sanity_1')

run_policy(env, get_action)
