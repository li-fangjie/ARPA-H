import numpy as np
import cv2 as cv
import logging
logging.basicConfig(level=logging.INFO)


leftExtrinsic = np.load("left.npy")
rightExtrinsic = np.load("right.npy")

def EstiamteEssentialMatrix(leftExtrinsic, rightExtrinsic):
    assert leftExtrinsic.shape[0] == rightExtrinsic.shape[1]

    
    extrinsics = [leftExtrinsic, rightExtrinsic]
    extrinsicMats = [np.zeros(leftExtrinsic.shape[0], 4, 4), np.zeros(leftExtrinsic.shape[0], 4, 4)]
    for iSide, extrinsic in enumerate(extrinsics):
        for i in range(extrinsic.shape[0]):
            curExtRVec, curExtTVec = extrinsic[0, 0, :, :], extrinsic[0, 0, :, :]
            curExtRMat = cv.Rodrigues(curExtRVec)
            extrinsicMats[iSide][i, :3, :3] = curExtRMat
            extrinsicMats[iSide][i, :3, 3] = curExtTVec
            extrinsicMats[iSide][i, 4, 4] = 1.0

    essentialMats = np.zeros(extrinsics[0].shape[0], 4, 4)
    for i in range(extrinsicMats[0].shape[0]):
        essentialMats[i, :, :] = np.linalg.inv(extrinsicMats[0][i, :, :]) @ extrinsicMats[1][i, :, :]
    return essentialMats

essentialMats = EstiamteEssentialMatrix(leftExtrinsic=leftExtrinsic, rightExtrinsic=rightExtrinsic)
np.save("essentialMats", essentialMats, allow_pickle=True)
    






