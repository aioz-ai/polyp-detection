import os 
import numpy as np 
import cv2
import pickle

IS_VIS = False
image_dir = "/mnt/ramdisk/isbi/datasets/CVC-ColonDB/CVC-ColonDB/jpg_original"
if not IS_VIS: txt_writer = open("./bboxes/cvc_colon_train_bboxes.txt",'w')
for ii in range(1,381):
    groundtruth_dir = os.path.join(image_dir,"p{}.jpg".format(ii))
    print(groundtruth_dir)
    # original_dir = os.path.join(train_dir,"j","{}.tif".format(ii))
    # original_img = cv2.imread(original_dir)
    image = cv2.imread(groundtruth_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(image,254,255,cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 30:
            continue
        if IS_VIS:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(image,(x,y),(x+w,y+h),color=235)
            cv2.imshow("IMAGE",image)
            cv2.waitKey(0)
            continue
        else:  
            x,y,w,h = cv2.boundingRect(contour)
            bbox = ["{}.jpg".format(ii),x,y,w,h]
            bbox = [str(elem) for elem in bbox]
            out_str = " ".join(bbox)
            print(out_str)
            txt_writer.writelines("{}\n".format(out_str))