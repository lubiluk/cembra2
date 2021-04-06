import os

import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from utils.extractors import CustomCNN
from utils.wrappers import BaselinifyWrapper

log_dir = "./data/reach_cam_sb_log"
save_path = "./data/reach_cam_sb"
best_save_path = "./data/reach_cam_sb_best"

os.makedirs(log_dir, exist_ok=True)

env = BaselinifyWrapper(
    TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
              max_episode_steps=100))

eval_env = BaselinifyWrapper(
    TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
              max_episode_steps=100))

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[64, 64],
                     features_extractor_class=CustomCNN,
                     features_extractor_kwargs=dict(features_dim=16,
                                                    linear_dim=16,
                                                    n_channels=1),
                     normalize_images=False)

model = SAC(
    "MlpPolicy",
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
env = TimeLimit(gym.make("PepperReachCam-v0", gui=True, dense=True),
                max_episode_steps=100)
model = SAC.load(save_path)
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
