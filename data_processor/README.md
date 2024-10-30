## Order of Operation
1. Save the video as a sequence of images
2. Frame-Sync them.
3. Given images of the checkerboard pattern, run `IntrinsicCalib.py` for each camera. It should output the intrinsic matrix and distortion of each camera.
4. Given the intrinsic calibration results, run `ExtrinsicsCalib.py` for them. It should output a series of estimated camera transformations.
5. Run `finishingExtrinsicCalib.ipynb` with the output from step 4. Verify that the error is in transformation is small and constant to time (otherwise, the frames are not properly synced). 