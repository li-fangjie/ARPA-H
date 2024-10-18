import numpy as np
import cv2 as cv
import sys

""" right
[[1.88239548e+03 0.00000000e+00 8.40338789e+02]
 [0.00000000e+00 1.87830380e+03 6.37196660e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]



[[-0.32437609  0.95671487  0.00645247  0.00263355 -0.75080285]]

"""

leftIntrinsic = {
    "Intrinsics": np.array(
        [[1.79975838e+03, 0.00000000e+00, 1.09477068e+03],
        [0.00000000e+00, 1.80696724e+03, 7.12522697e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "distortions": np.array(
        [[-0.29947173,  0.60303872, -0.00533951, -0.00191815, -0.29937121]],
    )
        }

rightIntrinsic = {
    "Intrinsics": np.array(
        [[1.88775988e+03, 0.00000000e+00, 8.41701545e+02],
        [0.00000000e+00, 1.88350574e+03, 6.36398991e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "distortions": np.array(
        [[-0.34163872, 1.1450937, 0.00621342, 0.002736, -1.28771471]],
    )
        }

curIntrinsic = rightIntrinsic
img = cv.imread(f"./Aug23_BPH_Stereo/checkerboard/mav0/cam1/data/1724435546364105984.png")
# img = cv.imread(f"./Aug23_BPH_Stereo/checkerboard/mav0/cam0/data/1724435546564293120.png")
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(curIntrinsic["Intrinsics"], curIntrinsic["distortions"], (w,h), 1, (w,h))

dst = cv.undistort(img, newcameramtx, curIntrinsic["distortions"], None, newcameramtx)

cv.imshow('img',dst)
cv.waitKey(0)

print(newcameramtx)
