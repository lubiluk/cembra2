from algos import ddpg_her
import torch as th
import gym
from utils import wrappers

def env_fn():
    return wrappers.DoneOnSuccessWrapper(gym.make('FetchPush-v1'))

ac_kwargs = dict(hidden_sizes=[64, 64], activation=th.nn.ReLU)

logger_kwargs = dict(
    output_dir='data/fetch_push_ddpg_her_1',
    exp_name='fetch_push_ddpg_her_1')

ddpg_her(
    env_fn=env_fn,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=10000,
    epochs=100,
    batch_size=256,
    replay_size=1000000,
    start_steps=0,
    update_after=0,
    gamma=0.95,
    q_lr=0.001,
    pi_lr=0.001,
    num_additional_goals=1,
    goal_selection_strategy='final',
    logger_kwargs=logger_kwargs,
    demos=[
        'data/demo_fetch_push/0.hdf5',
        'data/demo_fetch_push/1.hdf5',
        'data/demo_fetch_push/2.hdf5',
        'data/demo_fetch_push/3.hdf5',
        'data/demo_fetch_push/4.hdf5',
        'data/demo_fetch_push/5.hdf5',
        'data/demo_fetch_push/6.hdf5',
        'data/demo_fetch_push/7.hdf5',
        'data/demo_fetch_push/8.hdf5',
        'data/demo_fetch_push/9.hdf5'
    ])
