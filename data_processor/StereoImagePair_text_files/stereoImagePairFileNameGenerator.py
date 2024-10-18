import glob
import os

imagesLeft = glob.glob('/home/lifangjie/Desktop/Projects/APAR-H/Stereo_Aug23/Aug23_BPH_Stereo/checkerboard/mav0/cam0/data/*.png')
print(len(imagesLeft))

imagesRight = glob.glob('/home/lifangjie/Desktop/Projects/APAR-H/Stereo_Aug23/Aug23_BPH_Stereo/checkerboard/mav0/cam1/data/*.png')
print(imagesRight)

with open("./stereoImagePairs_3s.txt", 'w') as f:
    for i, (file1, file2) in enumerate(zip(imagesLeft, imagesRight)):
        if i % 4 == 0:
            f.write(f"{file1}\n")
            f.write(f"{file2}\n")

