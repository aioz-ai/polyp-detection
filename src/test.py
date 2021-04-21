import random
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
import os

from keras import *
from keras.models import *
from keras.layers import Input, Dense
from keras.preprocessing import image

from matplotlib.patches import Rectangle

WINDOW_SIZES = [150]      # using only one size for the sliding window
window_sizes=WINDOW_SIZES 
step=10                   # step of sliding on the input image (how to divide the original image)

img = skimage.io.imread('./testImage.tif', plugin='tifffile') # load a test image
print('Input image size=',img.shape[0],img.shape[1])
# plt.figure()
# plt.imshow(img)
# plt.show()

model = load_model('./transferVGG16_bottleneck_fc_model.h5')
print(model.summary())

max_pred = 0.0 # maximum prediction
max_box = []   # box for the polyp detection

print('--> Searching for a colonoscopy polyp ...')
# Loop window sizes: I will use only 150x150
for win_size in window_sizes:
    # Loop on both dimensions of the image
    for top in range(0, img.shape[0] - win_size + 1, step):
        for left in range(0, img.shape[1] - win_size + 1, step):
            # compute the (top, left, bottom, right) of the bounding box
            box = (top, left, top + win_size, left + win_size)

            # crop the original image
            cropped_img = img[box[0]:box[2], box[1]:box[3],:]
            
            # normalize the cropped image (the same processing used for the CNN dataset)
            cropped_img = cropped_img * 1./255
            # reshape from (150, 150, 3) to (1, 150, 150, 3) for prediction
            cropped_img = cropped_img.reshape((1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))

            # make a prediction for only one cropped small image 
            preds = model.predict(cropped_img, batch_size=None, verbose=0)
            # print(box[0],box[2],box[1],box[3], preds[0][0])
            if preds[0][0]> max_pred:
                max_pred = preds[0][0]
                max_box = box
print('Done!')

print('Best prediction:', max_box, max_pred)
plt.figure()
plt.imshow(img)
plt.text(1, -5, 'Best probability: '+str(max_pred), fontsize=10)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((max_box[1], max_box[0]), 150, 150,linewidth=1,edgecolor='r',facecolor='none'))
plt.show()