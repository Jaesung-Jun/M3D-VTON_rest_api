import os
import numpy as np
import cv2
"""
test = np.load(os.path.join("mpv3d_example", "depth", "BJ721E05W-J11@9=person_whole_back_depth.npy"))
print(np.max(test))
print(np.min(test))
cv2.imshow("test", test)
cv2.waitKey(0) 
cv2.DistroyAllWindows()
print(test.shape)
"""
test = cv2.imread(os.path.join("mpv3d_example", "image", "BJ721E05W-J11@9=person_whole_front.png"), 0)
streo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = streo.compute(test, test)
cv2.imshow("test", disparity)
cv2.waitKey(0) 
cv2.DistroyAllWindows()