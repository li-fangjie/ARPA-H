import numpy as np
import cv2
import glob

# # Define the dimensions of the checkerboard (number of inner corners per row and column)
# CHECKERBOARD = (10, 7)  # Adjust according to your checkerboard dimensions
# CHECKERBOARD_GRID_SIZE = 20 # mm
# images = glob.glob('/home/fj/Projects/ARPA-H/data/20241011_phantom_mono/run2/checkerboard/mav0/cam0/data/*.png')
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detectCheckerboardAndIntrinsicCalib(images, CHECKERBOARD_SHAPE=(10, 7), CHECKERBOARD_GRID_SIZE=20, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), savePath=None, imageRange=None, visualize=False):
    # ImageRange: [start, stop, finish]
    # 3D points in real world space
    objp = np.zeros((CHECKERBOARD_SHAPE[0]*CHECKERBOARD_SHAPE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SHAPE[0], 0:CHECKERBOARD_SHAPE[1]].T.reshape(-1, 2) * CHECKERBOARD_GRID_SIZE

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    import os
    # Get all the images from the folder
    # images = glob.glob('./data/20240823_phantom_stereo/checkerboard/mav0/cam1/data/*.png')
    for i, img_path in enumerate(images):
        print(f"\r{i}/{len(images)}", end="", flush=True)
        if imageRange != None:
            if i % imageRange[2] != 0:
                continue
            if i < imageRange[0]:
                continue
            if i > imageRange[1]:
                break
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SHAPE, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if visualize != False:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD_SHAPE, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    print("")
    cv2.destroyAllWindows()

    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print out the camera intrinsic matrix and distortion coefficients
    print("Error:\n", ret)
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)

    if not savePath is None:
        # Optionally, save the results to a file
        np.savez(savePath, ret=ret, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs
