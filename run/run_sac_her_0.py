from stable_baselines3 import HER
import gym
import gym_pepper

test_env = gym.make('PepperPush-v0', sim_steps_per_action=10, gui=True)
model = HER.load('./data/pepper_push_sac_her_1', env=test_env)

obs = test_env.reset()
for _ in range(100):
    action, _ = model.model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)

    if done:
        obs = test_env.reset()