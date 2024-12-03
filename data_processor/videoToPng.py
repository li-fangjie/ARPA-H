import cv2 as cv
import numpy as np
import glob
import logging
import os
logging.basicConfig(level=logging.INFO)

def saveFramesAsPng(videoPath, outputFolder, newSize=None, frameRange=None, timeStamps=None, saveTimeStampsFileName=None):
    # Ensure the output folder exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not timeStamps is None:
        timeStamps = timeStamps.astype(int)
    usedFrameTimes = []
    # Loop through each frame and save it as a PNG
    readIdx = 0
    idx = 0
    vidcap = cv.VideoCapture(videoPath)
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if success and (frameRange is None or idx < frameRange[1]):
            if frameRange is None or (len(frameRange) == 2 and idx >= frameRange[0] and idx < frameRange[1]) or (len(frameRange) == 3 and idx >= frameRange[0] and idx < frameRange[1] and (idx - frameRange[0]) % frameRange[2] == 0):
                if not newSize is None:
                    frame = cv.resize(frame, newSize)
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
                print(f"\r{readIdx}, {curTimeStamp}", end="")
                
                readIdx+=1
        else:
            break

        idx+=1
    
    usedFrameTimesArr = np.array(usedFrameTimes)
    if (saveTimeStampsFileName is None):
        saveTimeStampsFileName = f"{outputFolder}/time.csv"
    np.savetxt(saveTimeStampsFileName, usedFrameTimesArr, delimiter=",", fmt="%d")

def saveFramesAsPngSterescopic(leftVideoPath, rightVideoPath, leftOutputFolder, rightOutputFolder, leftTimeFile, rightTimeFile, newSize=None, offSetFrames=0, frameRange=None, timeStamps=None, saveTimeStampsFileName=None):
    raise NotImplementedError

if __name__ == "__main__":
    monoVideoBasePath = "/home/fj/Projects/ARPA-H/data/20241011_phantom_mono/run2/prostate/"
    monoVideoPath = f"{monoVideoBasePath}/Oct11_BPH_Recording3.avi"
    monoVideoOutputPath = f"{monoVideoBasePath}/mav0/cam0/data"
    leftTimeFilePath = f"{monoVideoBasePath}/Oct11_BPH_Recording3.csv"
    monoTimes = np.genfromtxt(leftTimeFilePath, delimiter=",", dtype=np.double)[:, 1]
    saveFramesAsPng(monoVideoPath, monoVideoOutputPath, frameRange=None, timeStamps=monoTimes, newSize=(640, 360))

    exit()


    offSetFrames = 6
    folderPath = "./data/20240823_phantom_stereo/prostate"
    leftOutputPath = f"{folderPath}/mav0/cam0/data"
    rightOutputPath = f"{folderPath}/mav0/cam1/data"
    leftTimeFilePath = f"{folderPath}/left_time.txt"
    rightTimeFilePath = f"{folderPath}/right_time.txt"

    leftVideoFilePath = f"{folderPath}/left.mp4"
    rightVideoFilePath = f"{folderPath}/right.mp4"

    logging.info("Reading Timestamps...")
    leftTimes = np.genfromtxt(leftTimeFilePath, delimiter=" ", dtype=np.double)
    rightTimes = np.genfromtxt(rightTimeFilePath, delimiter=" ", dtype=np.double)

    startFrame = 300 # 8940
    leftTimesCropped = leftTimes[offSetFrames+startFrame:, 1]
    rightTimesCropped = rightTimes[startFrame:, 1]
    minLen = min(leftTimesCropped.shape[0], rightTimesCropped.shape[0])

    frameRange = (startFrame, startFrame+minLen, 1)
    earlierFrameRange = (startFrame+offSetFrames, startFrame+offSetFrames+minLen, 1) # left
    logging.info(f"Left Video Frame Range: {earlierFrameRange}, Right Video Frame Range: {frameRange}")



    logging.info("Reading and saving left video frames...")
    saveFramesAsPng(leftVideoFilePath, leftOutputPath, frameRange=earlierFrameRange, timeStamps=leftTimesCropped, newSize=(640, 360))
    logging.info("Reading and saving right video frames...")
    saveFramesAsPng(rightVideoFilePath, rightOutputPath, frameRange=frameRange, timeStamps=leftTimesCropped, newSize=(640, 360))



# logging.info("Reading right video frames...")
# rightFrames = Utils.VideoToFrames(rightFilePath, frameRange)
# logging.info("Saving right video frames...")
# saveFramesAsPng(rightFrames, rightOutputPath, timeStamps=leftTimes)

