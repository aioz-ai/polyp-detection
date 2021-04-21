import os, cv2 
import pickle
import numpy as np 
import random
import time
import pandas as pd
import uuid
import multiprocessing
from collections import namedtuple
from data_aug import HorizontalFlip, Scale, Translate, Rotate, Shear

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    results = []
    for filename, x in zip(gb.groups.keys(), gb.groups):
        values = [filename]
        for index, row in gb.get_group(x).iterrows():
            xmin = row['xmin']
            xmax = row['xmax']
            ymin = row['ymin']
            ymax = row['ymax']
            values.extend([xmin,ymin,xmax,ymax])
        results.append(np.array(values))
    return results

def sequence_aug(group):
    # Init
    IMAGE_PRE_PATH = "/mnt/ramdisk/aug_data/CVC-ClinicDB/jpg_original"
    augmentations = [HorizontalFlip(), Scale(), Translate(), Rotate(random.uniform(0,360)), Shear()]
    probs = [0.2, 0.1, 0.1, 0.4, 0.1]
    try:
        # Process
        image_name = group[0]
        image_path = os.path.join(IMAGE_PRE_PATH,image_name)
        print(image_path)
        image = cv2.imread(image_path)
        bboxes = group[1:]
        bboxes = bboxes.reshape(-1,4)
        bboxes = np.hstack((bboxes,np.zeros((bboxes.shape[0],1)).astype(np.int32)))
        bboxes = np.stack(bboxes).astype(np.float64)

        for ii, augmentation in enumerate(augmentations):
            prob = probs[ii]            
            if random.random() < prob:
                image, bboxes = augmentation(image, bboxes)
        # Dump
        OUT_PATH = "/mnt/ramdisk/aug_data/augmented"
        image_out_name = str(uuid.uuid1())
        image_out_path = os.path.join(OUT_PATH,image_out_name+".jpg")
        cv2.imwrite(image_out_path,image)
        rs = {image_out_name:bboxes}
        return rs
    except Exception as exception:
        print(exception)
        return "None"
    

groundtruths = pd.read_csv("./polyp_train.csv")
filename_grouped = split(groundtruths,"filename")
filename_grouped = filename_grouped*100
pool = multiprocessing.Pool(processes=16)
results = pool.map(sequence_aug,filename_grouped)
pool.close()
pool.join()
print(results)
