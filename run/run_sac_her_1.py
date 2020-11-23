import gym
import gym_pepper
from gym.wrappers.time_limit import TimeLimit

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('./trained/pepper_push_sac_her_1')

env = TimeLimit(gym.make('PepperPush-v0', gui=True, sim_steps_per_action=10), max_episode_steps=300)

run_policy(env, get_action)