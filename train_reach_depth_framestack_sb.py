import os

import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from utils.extractors import CustomCNN
from utils.wrappers import DepthWrapper

log_dir = "./data/reach_depth_sb_log"
save_path = "./data/reach_depth_sb"
best_save_path = "./data/reach_depth_sb_best"

os.makedirs(log_dir, exist_ok=True)

def env_fn():
    return DepthWrapper(
    TimeLimit(gym.make("PepperReachDepth-v0", gui=False, dense=True),
              max_episode_steps=100))

env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=8, channels_order="first")

eval_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=8, channels_order="first")

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[64, 64, 64],
                     normalize_images=False)

model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=100_000,
    batch_size=512,
    learning_rate=0.0003,
    learning_starts=1000,
    gamma=0.95,
    ent_coef='auto',
    policy_kwargs=policy_kwargs,
    train_freq=1,
)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=best_save_path,
                             log_path=log_dir,
                             eval_freq=500,
                             deterministic=True,
                             render=False)

timesteps = 1_000_000

model.learn(timesteps, callback=eval_callback)

model.save(save_path)

# Evaluate
env.close()
env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=8, channels_order="first")
model = SAC.load(save_path)
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
