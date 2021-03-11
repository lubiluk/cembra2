import gym
import gym_pepper
from utils import wrappers
from gym.wrappers.time_limit import TimeLimit
from algos.test_policy import load_policy_and_env, run_policy

env = TimeLimit(gym.make("PepperReach-v0", gui=True, dense=True),
                max_episode_steps=100)

_, get_action = load_policy_and_env('trained/0_1')

run_policy(env, get_action)
