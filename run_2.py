import gym
from utils import wrappers
from utils.pepper_preprocessing import PepperPreprocessing
from gym.wrappers.time_limit import TimeLimit

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('./trained/2')

env = PepperPreprocessing(
    TimeLimit(gym.make("PepperReachCam-v0", gui=True), max_episode_steps=100))

run_policy(env, get_action)
