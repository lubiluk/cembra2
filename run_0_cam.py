import gym
import gym_pepper
from utils.wrappers import VisionWrapper, TorchifyWrapper
from gym.wrappers.time_limit import TimeLimit
from algos.test_policy import load_policy_and_env, run_policy

env = TimeLimit(VisionWrapper(
    TorchifyWrapper(gym.make("PepperReachCam-v0", gui=True, dense=True)),
    "trained/vision_0.pth"),
                max_episode_steps=100)

_, get_action = load_policy_and_env("trained/0")

run_policy(env, get_action)
