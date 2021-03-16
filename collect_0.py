import gym
import gym_pepper
import numpy as np
import cv2
import h5py
from algos.common.utils import combined_shape

N = 100000

env = gym.make("PepperReachCam-v0", gui=True)
obs = env.reset()

img_dim = env.observation_space.spaces["camera"].shape
lin_dim = env.observation_space.spaces["joints_state"].shape

with h5py.File("/scratch/collect_0.hdf5", "w") as f:
    img_dset = f.create_dataset("camera",
                                combined_shape(N, img_dim),
                                dtype='uint8')
    lin_dset = f.create_dataset("joints_state",
                                combined_shape(N, lin_dim),
                                dtype='f')
    pos_dset = f.create_dataset("object_position", (N, 3), dtype='f')

    for j in range(N):
        action = np.random.sample(10) * 2 - 1
        o, r, d, i = env.step(action)
        cv2.imshow("synthetic bottom camera", o["camera"])
        cv2.waitKey(1)

        img_dset[j] = o["camera"]
        lin_dset[j] = o["joints_state"]
        pos_dset[j] = i["object_position"]

        if d:
            env.reset()

    env.close()

