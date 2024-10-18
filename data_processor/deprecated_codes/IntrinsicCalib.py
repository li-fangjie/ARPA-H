import numpy as np
import cv2 as cv
import glob
import os
import logging
import Utils

logging.basicConfig(level=logging.INFO)

# CALIB PARAMS
folderPath = "./data/20240823_phantom_stereo/checkerboard"
fileName = "right.mp4"
filePath = f"{folderPath}/{fileName}"


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10, 7, 0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# TODO: Adapt MP4 read
# imgs = Utils.VideoToFrames(filePath, frameRange=(305, 1500, 3)) # glob.glob('*.jpg')
imgs = Utils.VideoToFrames(filePath, frameRange=(305, 1500,  5)) # glob.glob('*.jpg')
gray = cv.cvtColor(imgs[0], cv.COLOR_BGR2GRAY)

logging.info("Finding chessboard corners in video frames...")
for img in imgs:
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10, 7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (10,7), corners,ret)
        # cv.imshow('img',img)
        # cv.waitKey(100)

logging.info(f"All video frames processed! Chessboard corners found in {len(imgpoints)} images.")

cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret)
print("\n\n")
print(mtx)
print("\n\n")
print(dist)
# print("\n\n")
# print(rvecs)
# print("\n\n")
# print(tvecs)
# print("\n\n")


