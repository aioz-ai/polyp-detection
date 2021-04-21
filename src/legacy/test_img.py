import os 
import numpy as np 
import cv2
import pickle

train_dir = "/mnt/ramdisk/isbi/datasets/CVC-ClinicDB"
storage = np.array([],dtype=np.uint8)
for ii in range(1,613):
    groundtruth_dir = os.path.join(train_dir,"Ground Truth","{}.tif".format(ii))
    original_dir = os.path.join(train_dir,"Original","{}.tif".format(ii))
    original_img = cv2.imread(original_dir)
    image = cv2.imread(groundtruth_dir,cv2.IMREAD_UNCHANGED)
#     ret, binary_img = cv2.threshold(image,255,255,cv2.THRESH_BINARY)
    whites = np.where(image==255)
    image[whites] = 0
    counter = np.where(image>0)[0].shape[0]
    if counter > 0:
        storage = np.append(storage,image[np.where(image>0)])
print(storage)
print(np.amin(storage))
print(np.amax(storage))
uniqueValues = np.unique(storage)
print(uniqueValues)