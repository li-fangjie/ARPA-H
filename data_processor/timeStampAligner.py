import os
import argparse
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import csv

def matrixToTranslationQuaternion(matrix, returnSingle=True):
    """Convert a 4x4 transformation matrix to translation vector and quaternion."""
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 transformation matrix.")

    translation = matrix[:3, 3]  # The last column (first three elements)

    rotationMatrix = matrix[:3, :3]

    rotation = R.from_matrix(rotationMatrix)
    quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]

    if returnSingle:
        return np.concatenate((translation, quaternion))
    return translation, quaternion

def loadOpticalTrackerData(filePath):
    """Load optical tracker data from a CSV file."""
    # data = np.loadtxt(filePath, delimiter=',')
    # timestamps = data[:, 0]  # Assuming the first column is the timestamp
    # transformations = data[:, 1:]  # The remaining columns are the 6 DoF data

    with open(filePath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        dataReadList = []
        timestamps = []
        transforms = []
        for row in reader:
            rowRead = []
            for element in row:    
                # elementCleaned = element.replace("array", "").replace("(", "").replace(")", "").replace("nan", "float('nan')").replace("\n", "")
                elementCleaned = element.replace("array(", "").replace(")", "").replace("[", "").replace("]", "").replace("\n", "")
                dataElement = np.fromstring(elementCleaned, sep=",")
                if dataElement.shape[0] % 16 == 0 and dataElement.shape[0] >= 16:
                    dataElement = dataElement.reshape([dataElement.shape[0] // 16, 4, 4])
                rowRead.append(dataElement)
            timestamps.append(rowRead[1])
            dataReadList.append(rowRead)
            transforms.append(rowRead[3])
        timestamps = np.array(timestamps)
        transforms = np.array(transforms)
    return timestamps, transforms

def loadVideoTimestamps(filePath, **args):
    """Load video timestamps from a CSV file."""
    return np.loadtxt(filePath, delimiter=',', **args)

def processDirectory(directory):
    """Process the given directory to find and align the data."""
    # Find files in the directory
    ndiFiles = [f for f in os.listdir(directory) if f.startswith("ndi") and f.endswith(".csv") and not f.endswith("original.csv") and not f.endswith("aligned.csv")]
    videoFiles = [f for f in os.listdir(directory) if f.startswith("endoscope_timestamp_") and f.endswith(".csv")]

    # Load the data from the files
    if not ndiFiles or not videoFiles:
        print("No appropriate files found in the directory.")
        return
    print(f"Found {len(ndiFiles)} NDI files and {len(videoFiles)} video timestamp files.")
    
    # trackerRawDataList = []
    # for ndiFileI in range(len(ndiFiles)):
    #     # Assuming only one file each, otherwise this can be extended to handle multiple files
    #     ndiFilePath = os.path.join(directory, ndiFiles[ndiFileI])
    #     trackerTimestamps, trackerTransformData = loadOpticalTrackerData(ndiFilePath)
    #     trackerTimestamps *= 1000
    #     trackerRawDataList.append([trackerTimestamps, trackerTransformData])

    # videoRawDataList = []
    # for vidoeFileI in range(len(videoFiles)):
    #     videoFilePath = os.path.join(directory, videoFiles[vidoeFileI])
    #     videoTimestamps = loadVideoTimestamps(videoFilePath, usecols=[-1])
    #     videoRawDataList.append(videoTimestamps)


    for ndiFileI in range(len(ndiFiles)):
        # Assuming only ONE VIDEO FILE, otherwise this can be extended to handle multiple files
        ndiFilePath = os.path.join(directory, ndiFiles[ndiFileI])
        videoFilePath = os.path.join(directory, videoFiles[0])
        outputFileName = ndiFiles[ndiFileI].split(".csv")[0] + "_aligned.csv"

        print(f"Processing NDI file: {ndiFilePath}")
        print(f"Processing video timestamp file: {videoFilePath}")

        # Load the data
        # trackerTransformData: nTime x nTool x 4 x 4
        trackerTimestamps, trackerTransformData = loadOpticalTrackerData(ndiFilePath)
        trackerTimestamps *= 1000
        videoTimestamps = loadVideoTimestamps(videoFilePath, usecols=[-1])

        print(trackerTransformData.shape)
        trackerData = []
        for timeI in range(trackerTransformData.shape[0]):
            curTimeData = []
            for toolI in range(trackerTransformData.shape[1]):
                curTimeData.append(matrixToTranslationQuaternion(trackerTransformData[timeI, toolI, :, :].squeeze()))
            trackerData.append(curTimeData)
        trackerData = np.array(trackerData)
        # Interpolate the tracker data for each video timestamp

        interpolatedData = np.zeros((len(videoTimestamps),  trackerData.shape[1], trackerData.shape[2]))
        for toolI in range(trackerData.shape[1]):
            for elementI in range(trackerData.shape[2]):
                # Interpolate each of the 6 DoF transformations
                interpFunc = interp1d(trackerTimestamps[:, toolI], trackerData[:, toolI, elementI], kind='linear', fill_value="extrapolate")
                interpolatedData[:, toolI, elementI] = interpFunc(videoTimestamps)
        
        interpolatedData = interpolatedData.reshape([interpolatedData.shape[0], -1])

        # Output the aligned data
        outputFile = os.path.join(directory, outputFileName)
        np.savetxt(outputFile, np.hstack((videoTimestamps[:, None], interpolatedData)), delimiter=',',
                header='', comments='')

        print(f"Aligned data saved to {outputFile}")

if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Align optical tracking data with video timestamps.")
    parser.add_argument("--data", type=str, help="Directory containing the CSV files.")

    # Parse the arguments
    args = parser.parse_args()

    # Process the directory
    processDirectory(args.data)
