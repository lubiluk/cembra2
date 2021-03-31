import gym
import gym_pepper
import numpy as np
import cv2
import h5py
from algos.common.utils import combined_shape
from pathlib import Path

DST_HDF = "data/collect_0.hdf5"
DST_DIR = "data/collect_0"

dir = Path(DST_DIR)

dir.mkdir(exist_ok=True)

# Clear old data from dir if exists
for child in dir.glob('*'):
    child.unlink()

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
    cpos_dset = f.create_dataset("camera_pose", (N, 7), dtype='f')

    for j in range(N):
        action = np.random.sample(10) * 2 - 1
        o, r, d, i = env.step(action)

        img = o["camera"]
        lin = o["joints_state"]
        pos = i["object_position"]
        cpos = o["camera_pose"]

        cv2.imshow("camera", img)
        cv2.imwrite(
            DST_DIR + "/{}_{}_{}_{}.png".format(j, pos[0], pos[1], pos[2]),
            o["camera"])
        cv2.waitKey(1)

        img_dset[j] = img
        lin_dset[j] = lin
        pos_dset[j] = pos
        cpos_dset[j] = cpos

        if d:
            env.reset()

    env.close()
