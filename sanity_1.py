# Sanity test on Fetch
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import td3
from algos.td3 import core


def env_fn():
    return gym.make('Pendulum-v0')


ac_kwargs = dict(hidden_sizes=[400, 300], activation=nn.ReLU)

logger_kwargs = dict(output_dir='data/sanity_1', exp_name='sanity_1')

td3(env_fn=env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=1000,
    max_ep_len=1000,
    epochs=10,
    batch_size=100,
    replay_size=1000000,
    gamma=0.99,
    pi_lr=0.001,
    q_lr=0.001,
    polyak=0.995,
    update_after=100,
    update_every=1,
    logger_kwargs=logger_kwargs)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/sanity_1')

env = gym.make('Pendulum-v0')

run_policy(env, get_action)
