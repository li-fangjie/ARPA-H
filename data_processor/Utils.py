import numpy as np
import cv2 as cv
import glob
import os
import logging


def VideoToFrames(videoPath, newSize=None, frameRange=None):
    vidcap = cv.VideoCapture(videoPath)
    imgs = []
    itr = 0
    readCount = 0
    logging.info(f"Reading video file {videoPath}...")
    while vidcap.isOpened():
        success, image = vidcap.read()
        if not newSize is None:
            image = cv.resize(image, newSize)
        itr += 1
        if success and (frameRange is None or itr < frameRange[1]):
            if frameRange is None or (len(frameRange) == 2 and itr >= frameRange[0] and itr < frameRange[1]) or (len(frameRange) == 3 and itr >= frameRange[0] and itr < frameRange[1] and (itr - frameRange[0]) % frameRange[2] == 0):
                imgs.append(image)
                readCount+=1
        else:
            break
    cv.destroyAllWindows()
    vidcap.release()
    logging.info(f"Video file read! {readCount} frames read.")
    return imgs

def frameTimeComparison(timePath, frameRange=None):
    return
    
