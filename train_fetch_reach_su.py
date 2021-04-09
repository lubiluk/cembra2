# Sanity test on Fetch
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core_her
from algos.common import replay_buffer_her

env = gym.make("FetchReach-v1")

ac_kwargs = dict(hidden_sizes=[64, 64], activation=nn.ReLU)
rb_kwargs = dict(size=1000000,
                 n_sampled_goal=4,
                 goal_selection_strategy='future')

logger_kwargs = dict(output_dir='data/fetch_reach_su', exp_name='fetch_reach_su')

model = SAC(env=env,
    actor_critic=core_her.MLPActorCritic,
    ac_kwargs=ac_kwargs,
    replay_buffer=replay_buffer_her.ReplayBuffer,
    rb_kwargs=rb_kwargs,
    max_ep_len=100,
    batch_size=256,
    gamma=0.95,
    lr=0.001,
    ent_coef="auto",
    update_after=1000,
    update_every=1,
    logger_kwargs=logger_kwargs)

model.train(steps_per_epoch=3000, epochs=10)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/fetch_reach_su', deterministic=True)

run_policy(env, get_action)