import gym
from utils import wrappers

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('./trained/fetch_push_sac_her_3')

env = wrappers.DoneOnSuccessWrapper(gym.make('FetchPush-v1'))

run_policy(env, get_action)