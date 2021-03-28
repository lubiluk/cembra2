import gym
import gym_pepper
import numpy as np
import cv2
import h5py
from algos.common.utils import combined_shape

DST_HDF = "data/collect_0.hdf5"
DST_DIR = "data/collect_0"

N = 100000

env = gym.make("PepperReachCam-v0", gui=False)
obs = env.reset()

img_dim = env.observation_space.spaces["camera"].shape
lin_dim = env.observation_space.spaces["joints_state"].shape

lower_range = np.array([110, 50, 50])
upper_range = np.array([130, 255, 255])

with h5py.File(DST_HDF, "w") as f:
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

        img = o["camera"]
        lin = o["joints_state"]
        pos = i["object_position"]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_range, upper_range)

        cv2.imshow("camera", img)
        cv2.imshow('mask', mask)

        if mask.max() == 0:
            pos = np.array([0, 0, 0], dtype=np.float32)

        cv2.imwrite(
            "data/collect_0/{}_{}_{}_{}.png".format(
                j, pos[0], pos[1], pos[2]), o["camera"])
        cv2.waitKey(1)

        img_dset[j] = o["camera"]
        lin_dset[j] = o["joints_state"]
        pos_dset[j] = i["object_position"]

        if d:
            env.reset()

    env.close()
