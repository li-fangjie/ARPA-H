import numpy as np
import cv2 as cv
import glob
import os
import logging
import Utils
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

# leftIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.81691143e+03, 0.00000000e+00, 1.08824794e+03],
#          [0.00000000e+00, 1.82476479e+03, 7.12502755e+02],
#          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#     "distortions": np.array(
#         [[-0.31365749,  0.68343923, -0.00557816, -0.00154004, -0.37446887]]    
#         )
#         }


# 1.455601689292678




# LeftIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.79975838e+03, 0.00000000e+00, 1.09477068e+03],
#         [0.00000000e+00, 1.80696724e+03, 7.12522697e+02],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#     "distortions": np.array(
#         [[-0.29947173,  0.60303872, -0.00533951, -0.00191815, -0.29937121]],
#     )
#         }

# rightIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.88239548e+03, 0.00000000e+00, 8.40338789e+02,],
#         [0.00000000e+00, 1.87830380e+03, 6.37196660e+02,],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00,]]),
#     "distortions": np.array(
#     [[-0.32437609,  0.95671487,  0.00645247,  0.00263355, -0.75080285]]
#     )
#         }

# leftIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.83395055e+03, 0.00000000e+00, 1.08960188e+03],
#          [0.00000000e+00, 1.84166490e+03, 7.11227520e+02],
#          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]),
#     "distortions": np.array(
#          [[-0.3193637, 0.72440511, -0.00520537, -0.00134968, -0.43997669]]
#     )
#         }
# 1.3176426936669425
# leftIntrinsic = {
#     "Intrinsics": np.array(       
#         [[1.78550748e+03, 0.00000000e+00, 1.09446159e+03,],
#         [0.00000000e+00, 1.79280911e+03, 7.15388277e+02,],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00,]]),
#     "distortions": np.array(
#          [[-0.30015264,  0.61177836, -0.0060459,  -0.00196304, -0.31251207]]
#     )
#         }
rightIntrinsic = {
    "Intrinsics": np.array(       
        [[628.43199339,   0.,         279.8172176 ],
        [  0.,         627.01510146, 210.78945537],
        [  0.,           0.,           1.        ],]),
    "distortions": np.array(
        [[-0.32994704,  0.99665627,  0.00600938,  0.00255975, -0.86428543]]
    )
        }
# rightIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.88775988e+03, 0.00000000e+00, 8.41701545e+02],
#         [0.00000000e+00, 1.88350574e+03, 6.36398991e+02],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#     "distortions": np.array(
#         [[-0.34163872, 1.1450937, 0.00621342, 0.002736, -1.28771471]],
#     )
#         }

# rightIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.87303773e+03, 0.00000000e+00, 8.41790761e+02],
#          [0.00000000e+00, 1.86947169e+03, 6.39315479e+02],
#          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]),
#     "distortions": np.array(
#          [[-0.32686964,  1.01910631,  0.00525282,  0.00268762, -0.97685181]]
#     )
#         }

# rightIntrinsic = {
#     "Intrinsics": np.array(
#         [[1.89286241e+03, 0.00000000e+00, 8.40067071e+02],
#         [0.00000000e+00, 1.88868641e+03, 6.35149799e+02],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]),
#     "distortions": np.array(
#          [[-0.34145388,  1.16739137,  0.00630022,  0.00262966, -1.42688215]]
#     )
#         }
leftIntrinsic = {
    "Intrinsics": np.array(
        [[616.25592128,   0.,         361.65777105],
        [  0.,         618.5552621,  233.81524659],
        [  0.,           0.,           1.        ]]),
    "distortions": np.array(
        [[-0.31521841,  0.64575692, -0.00443106, -0.00141811, -0.20538541]]    )
        }

# 1.1860641777859393



folderPath = "./data/20240823_phantom_stereo/checkerboard"

leftFilePath = os.path.join(folderPath, "mav0", "cam0", "data") # f"{folderPath}/left.mp4"
rightFilePath = os.path.join(folderPath, "mav0", "cam1", "data") # f"{folderPath}/right.mp4"
leftTimeFilePath = f"{folderPath}/left_time.txt"
rightTimeFilePath = f"{folderPath}/right_time.txt"
frameRange = (300, 1843, 1)
earlierFrameRange = (300+6, 1843, 1)
boardShape = (10, 7)

class StereoCameraPair:
    def __init__(self, name, laterFrameCamera, earlierFrameCamera, frameOffset=0):
        self.name = name
        self.laterFrameCamera = laterFrameCamera
        self.earlierFrameCamera = earlierFrameCamera
        self.frameOffset = frameOffset
        self.allCameras = [self.earlierFrameCamera, self.laterFrameCamera]

class VideoCameraData:
    def __init__(self, name, cameraData, filePath, timeFilePath, frameRange, boardShape, ):
        self.name = name
        self.cameraData = cameraData
        self.filePath = filePath
        self.timeFilePath = timeFilePath
        self.frameRange = frameRange
        self.boardShape = boardShape

        if filePath.endswith("mp4"):
            self.imgs = Utils.VideoToFrames(self.filePath, frameRange=self.frameRange)
        else:
            imgFileNames = glob.glob(os.path.join(filePath, "*.png"))
            self.imgs = []
            for imgFileName in imgFileNames:
                self.imgs.append(cv.imread(imgFileName))
        self.frameTimes = np.genfromtxt(self.timeFilePath, delimiter=" ")
        return


leftData = VideoCameraData("left", leftIntrinsic, leftFilePath, leftTimeFilePath, earlierFrameRange, boardShape)
rightData = VideoCameraData("right", rightIntrinsic, rightFilePath, rightTimeFilePath, frameRange, boardShape)
cameraPair = StereoCameraPair(name="cameraPairExtrinsic", laterFrameCamera=rightData, earlierFrameCamera=leftData, frameOffset=0)
outputFileName = f"{cameraPair.name}_{timestamp}"

allDatas = [leftData, rightData]

curData = rightData
leftImgs = leftData.imgs
rightImgs = rightData.imgs


print(len(leftImgs))
print(len(rightImgs))
allImgs = [leftImgs, rightImgs]


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((curData.boardShape[0] * curData.boardShape[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:curData.boardShape[0],0:curData.boardShape[1]].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

extrinsics = [[],[]]
# for i in range(0, min(cameraPair.earlierFrameCamera.frameTimes.shape[0] - cameraPair.frameOffset, cameraPair.laterFrameCamera.frameTimes.shape[0])):
totalFrameNum = min(len(cameraPair.earlierFrameCamera.imgs), len(cameraPair.laterFrameCamera.imgs))
for i in range(0, totalFrameNum):
    print(f"{i}/{totalFrameNum}")
    ret = True
    corners = []
    grays = []
    imgs = []

    for jData, curData in enumerate(cameraPair.allCameras):
        curImg = cameraPair.allCameras[jData].imgs[i]
        imgs.append(curImg)
        curGray = cv.cvtColor(curImg,cv.COLOR_BGR2GRAY)
        grays.append(curGray)
        curRet, curCorners = cv.findChessboardCorners(curGray, curData.boardShape, None)
        corners.append(curCorners)
        ret = (ret and curRet)
    if ret:
        curCorners2s = []
        imgPtsList = []
        for jData, curData in enumerate(allDatas):
            curCorners2 = cv.cornerSubPix(grays[jData],corners[jData],(11,11),(-1,-1),criteria)
    
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, curCorners2, curData.cameraData["Intrinsics"], curData.cameraData["distortions"])
            extrinsics[jData].append((rvecs, tvecs))

            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, curData.cameraData["Intrinsics"], curData.cameraData["distortions"])

            curCorners2s.append(curCorners2)
            imgPtsList.append(imgpts)

        # imgsDrawn = []
        # for jData, img in enumerate(imgs):
        #     imgsDrawn.append(draw(img, curCorners2s[jData], imgPtsList[jData]))
        # combinedImg = np.hstack(imgsDrawn)
        # cv.imshow('img',combinedImg)
        # k = cv.waitKey(20) & 0xFF
        """ k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(f'{i}.png', img) """
    
cv.destroyAllWindows()


np.save(outputFileName, extrinsics, allow_pickle=True)

