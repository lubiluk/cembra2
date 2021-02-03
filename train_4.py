# Sanity test on Fetch
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import td3_her
from algos.td3_her import core
from gym.wrappers.time_limit import TimeLimit


def env_fn():
    return gym.make("FetchPush-v1", gui=False)


ac_kwargs = dict(hidden_sizes=[128, 128], activation=nn.ReLU)

logger_kwargs = dict(output_dir='data/1', exp_name='1')

td3_her(env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=10000,
        max_ep_len=100,
        epochs=100,
        batch_size=64,
        replay_size=100000,
        gamma=0.95,
        pi_lr=1e-3,
        q_lr=1e-3,
        update_after=1000,
        update_every=64,
        num_additional_goals=4,
        goal_selection_strategy='future',
        logger_kwargs=logger_kwargs)
