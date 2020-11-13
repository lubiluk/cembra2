runninfrom algos.test_policy import load_policy_and_env, run_policy
import gym
import gym_pepper

_, get_action = load_policy_and_env('./data/pepper_push_ddpg_her_0')

env = gym.make('PepperPush-v0', gui=True, sim_steps_per_action=10)
env.goal = "yea"

run_policy(env, get_action)