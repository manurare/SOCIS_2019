from os import listdir
from os.path import isfile, join
import cv2
path = '/home/manuelrey/ESA/Dataset/Step2-SuperresolutionWhale/converted'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
for file in onlyfiles:
    img = cv2.imread(join(path,file))
    if img.shape[2] != 3:
        print("EO")


