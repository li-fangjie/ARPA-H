import numpy as np
import cv2 as cv
import glob
import os
import logging
import Utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


folderPath = "../data/20240823_phantom_stereo/prostate"

class VideoCameraData:
    def __init__(self, name, cameraData, filePath, timeFilePath, frameRange, boardShape, readVideoNow=False):
        self.name = name
        self.cameraData = cameraData
        self.filePath = filePath
        self.timeFilePath = timeFilePath
        self.frameRange = frameRange
        self.boardShape = boardShape
        
        if readVideoNow:
            if filePath.endswith("mp4"):
                self.imgs = Utils.VideoToFrames(self.filePath, frameRange=self.frameRange)
            else:
                imgFileNames = glob.glob(os.path.join(filePath, "*.png"))
                self.imgs = []
                for imgFileName in imgFileNames:
                    self.imgs.append(cv.imread(imgFileName))
            self.frameTimes = np.genfromtxt(self.timeFilePath, delimiter=" ")
        return

        

leftTimeFilePath = f"{folderPath}/left_time.txt"
rightTimeFilePath = f"{folderPath}/right_time.txt"
leftVideoFilePath = f"{folderPath}/left.mp4"
rightVideoFilePath = f"{folderPath}/right.mp4"

leftOutputPath = f"{folderPath}/mav0/cam1/data"
rightOutputPath = f"{folderPath}/mav0/cam0/data"

## Time Frame Alignment
leftTimes = np.genfromtxt(leftTimeFilePath, delimiter=" ", dtype=np.double)
rightTimes = np.genfromtxt(rightTimeFilePath, delimiter=" ", dtype=np.double)
if leftTimes[0, 1] <= rightTimes[0, 1]:
    earlierStartTime = leftTimes
    laterStartTime = rightTimes
    print("left camera started earlier")
else:
    earlierStartTime = rightTimes
    laterStartTime = leftTimes
    print("right camera started earlier")

curMinOffset = np.inf
curMinIdx = 0
for i in range(earlierStartTime.shape[0]):
    tDiff = (laterStartTime[0, 1] - earlierStartTime[i, 1]) * 1e-3 # micro s
    if (abs(tDiff) < curMinOffset):
        curMinOffset = tDiff
        curMinIdx = i

print(f"earlier start frame idx {curMinIdx}, frame # {earlierStartTime[curMinIdx, 0]}, {curMinOffset}")

tOffset = 6
tDiffs = []
for i in range(0, min(earlierStartTime.shape[0] - tOffset, laterStartTime.shape[0])):
    tDiff = (laterStartTime[i, 1] - earlierStartTime[i + tOffset, 1]) * 1e-3 # micro s
    tDiffs.append(abs(tDiff))

print("Mean Frame Time Difference (us)", np.mean(tDiffs))
print("Max Frame Time Difference (us)", np.max(tDiffs))


## Save Video as Frames
def saveFramesAsPng(videoPath, outputFolder, newSize=None, frameRange=None, timeStamps=None, saveTimeStampsFileName=None):
    # Ensure the output folder exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    timeStamps = timeStamps.astype(np.int64)
    usedFrameTimes = []
    # Loop through each frame and save it as a PNG
    readIdx = 0
    idx = 0
    vidcap = cv.VideoCapture(videoPath)
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not newSize is None:
            frame = cv.resize(frame, newSize)
        if success and (frameRange is None or idx < frameRange[1]):
            if frameRange is None or (len(frameRange) == 2 and idx >= frameRange[0] and idx < frameRange[1]) or (len(frameRange) == 3 and idx >= frameRange[0] and idx < frameRange[1] and (idx - frameRange[0]) % frameRange[2] == 0):

                # Create a unique filename
                if timeStamps is None:
                    curTimeStamp = readIdx
                else:
                    curTimeStamp = timeStamps[readIdx]

                fileName = f"{curTimeStamp:d}.png"
                filename = os.path.join(outputFolder, fileName)
                # Save the frame as a PNG
                cv.imwrite(filename, frame)
                usedFrameTimes.append(curTimeStamp)
                print(readIdx, end="")
                readIdx+=1
        else:
            break

        idx+=1
    
    usedFrameTimesArr = np.array(usedFrameTimes)
    if (saveTimeStampsFileName is None):
        saveTimeStampsFileName = f"{outputFolder}/time.csv"
    np.savetxt(saveTimeStampsFileName, usedFrameTimesArr, delimiter=",", fmt="%d")

logging.info("Reading Timestamps...")
leftTimes = np.genfromtxt(leftTimeFilePath, delimiter=" ", dtype=np.double)
rightTimes = np.genfromtxt(rightTimeFilePath, delimiter=" ", dtype=np.double)

startFrame = 300 # 8940
leftTimesCropped = leftTimes[curMinIdx+startFrame:, 1]
rightTimesCropped = rightTimes[startFrame:, 1]
minLen = min(leftTimesCropped.shape[0], rightTimesCropped.shape[0])

frameRange = (startFrame, startFrame+minLen, 1)
earlierFrameRange = (startFrame+curMinIdx, startFrame+curMinIdx+minLen, 1) # left
logging.info(f"Left Video Frame Range: {earlierFrameRange}, Right Video Frame Range: {frameRange}")



logging.info("Reading and saving left video frames...")
saveFramesAsPng(leftVideoFilePath, leftOutputPath, frameRange=earlierFrameRange, timeStamps=leftTimesCropped, newSize=(640, 360))
logging.info("Reading and saving right video frames...")
saveFramesAsPng(rightVideoFilePath, rightOutputPath, frameRange=frameRange, timeStamps=leftTimesCropped, newSize=(640, 360))

