import numpy as np

data = np.load("mpv3d_example/depth/BJ721E05W-J11@9=person_whole_front_depth.npy")
print(data.max())
print(data.shape)