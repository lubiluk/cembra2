# Sanity test on Fetch
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import sac
from algos.sac import core_her
from algos.common import her_replay_buffer

env = gym.make("FetchReach-v1")

ac_kwargs = dict(hidden_sizes=[64, 64], activation=nn.ReLU)
rb_kwargs = dict(size=1000000,
                 n_sampled_goal=4,
                 goal_selection_strategy='future')

logger_kwargs = dict(output_dir='data/sanity_2', exp_name='sanity_2')

sac(env=env,
    actor_critic=core_her.MLPActorCritic,
    ac_kwargs=ac_kwargs,
    replay_buffer=her_replay_buffer.HerReplayBuffer,
    rb_kwargs=rb_kwargs,
    steps_per_epoch=10000,
    max_ep_len=10000,
    epochs=20,
    batch_size=256,
    gamma=0.95,
    lr=0.001,
    update_after=1000,
    update_every=1,
    logger_kwargs=logger_kwargs)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/sanity_2')

run_policy(env, get_action)