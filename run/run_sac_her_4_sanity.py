from stable_baselines3 import HER
import gym
from utils import wrappers
from algos.core import combined_shape
import numpy as np
import h5py

test_env =  wrappers.DoneOnSuccessWrapper(gym.make('FetchPush-v1'))
model = HER.load('./trained/fetch_push_sac_her_4', env=test_env)

obs_dim = test_env.observation_space.spaces["observation"].shape
act_dim = test_env.action_space.shape[0]
goal_dim = test_env.observation_space.spaces["desired_goal"].shape

ep = 0
obs_buf = np.zeros(combined_shape(1000, obs_dim))
obs2_buf = np.zeros(combined_shape(1000, obs_dim))
act_buf = np.zeros(combined_shape(1000, act_dim))
rew_buf = np.zeros(1000)
done_buf = np.zeros(1000)
dgoal_buf = np.zeros(combined_shape(1000, goal_dim))
agoal_buf = np.zeros(combined_shape(1000, goal_dim))
ptr = 0

obs = test_env.reset()
for _ in range(1000):
    test_env.render()
    action, _ = model.model.predict(obs, deterministic=True)
    
    obs_buf[ptr] = obs['observation']

    obs, reward, done, _ = test_env.step(action)
    
    obs2_buf[ptr] = obs['observation']
    act_buf[ptr] = action
    rew_buf[ptr] = reward
    done_buf[ptr] = done
    dgoal_buf[ptr] = obs['desired_goal']
    agoal_buf[ptr] = obs['achieved_goal']
    ptr += 1


    if done:
        fp = "data/demo_fetch_push/{}.hdf5".format(ep)
        size = ptr

        with h5py.File(fp, "w") as f:
            obs = f.create_dataset("obs", combined_shape(size, obs_dim), dtype='f')
            obs2 = f.create_dataset("obs2", combined_shape(size, obs_dim), dtype='f')
            act = f.create_dataset("act", combined_shape(size, act_dim), dtype='f')
            rew = f.create_dataset("rew", (size, ), dtype='f')
            done = f.create_dataset("done", (size, ), dtype='f')
            dgoal = f.create_dataset("dgoal", combined_shape(size, goal_dim), dtype='f')
            agoal = f.create_dataset("agoal", combined_shape(size, goal_dim), dtype='f')

            obs[...] = obs_buf[:size]
            obs2[...] = obs2_buf[:size]
            act[...] = act_buf[:size]
            rew[...] = rew_buf[:size]
            done[...] = done_buf[:size]
            dgoal[...] = dgoal_buf[:size]
            agoal[...] = agoal_buf[:size]

        obs = test_env.reset()
        obs_buf = np.zeros(combined_shape(1000, obs_dim))
        obs2_buf = np.zeros(combined_shape(1000, obs_dim))
        act_buf = np.zeros(combined_shape(1000, act_dim))
        rew_buf = np.zeros(1000)
        done_buf = np.zeros(1000)
        dgoal_buf = np.zeros(combined_shape(1000, goal_dim))
        agoal_buf = np.zeros(combined_shape(1000, goal_dim))
        ptr = 0
        ep += 1

