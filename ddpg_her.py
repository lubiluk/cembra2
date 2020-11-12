from algos import ddpg_her
import torch as th
import gym
import gym_pepper


def env_fn(): return gym.make('PepperPush-v0')


ac_kwargs = dict(hidden_sizes=[128, 128], activation=th.nn.ReLU)

logger_kwargs = dict(
    output_dir='data/pepper_push_ddpg_her_0',
    exp_name='pepper_push_ddpg_her_0')

ddpg_her(
    env_fn=env_fn,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=15000,
    epochs=200,
    batch_size=256,
    replay_size=1000000,
    start_steps=0,
    update_after=0,
    gamma=0.95,
    q_lr=0.001,
    pi_lr=0.001,
    num_additional_goals=4,
    goal_selection_strategy='future',
    logger_kwargs=logger_kwargs)