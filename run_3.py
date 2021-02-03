import gym
from utils import wrappers
from gym.wrappers.time_limit import TimeLimit

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('./data/3')

env = TimeLimit(gym.make("PepperPushCam-v0", gui=True, dense=True),
                     max_episode_steps=100)

run_policy(env, get_action)
