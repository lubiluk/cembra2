# Sanity test on Fetch
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import td3_her
from algos.td3_her import core
from utils.wrappers import DoneOnSuccessWrapper


def env_fn():
    return DoneOnSuccessWrapper(gym.make("FetchPush-v1"))


ac_kwargs = dict(hidden_sizes=[64, 64], activation=nn.ReLU)

logger_kwargs = dict(output_dir='data/sanity_2', exp_name='sanity_2')

td3_her(env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=15000,
        max_ep_len=300,
        epochs=100,
        batch_size=256,
        replay_size=1000000,
        gamma=0.95,
        pi_lr=0.001,
        q_lr=0.001,
        update_after=1000,
        update_every=1,
        num_additional_goals=4,
        goal_selection_strategy='future',
        logger_kwargs=logger_kwargs)



from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/sanity_2')

env = env_fn()

run_policy(env, get_action)