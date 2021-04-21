import os
import numpy as np
import sys
import cv2
import time
import zipfile
from PIL import Image
import pandas as pd
import tensorflow as tf

from polyp_utils import split
from polyp_utils import load_model
from polyp_utils import load_image_into_numpy_array
from polyp_utils import polyp_evaluation_metric
from polyp_utils import parse_predictions, parse_groundtruths
from polyp_utils import cal_cfs_matrix

PATH_TO_FROZEN_GRAPH = "models/polyp_detection_autoaugment_bestmodel.pb"
TEST_DIR = 'test_images'

detection_graph = load_model(PATH_TO_FROZEN_GRAPH)

groundtruths = pd.read_csv("./polyp_test.csv")
filename_grouped = split(groundtruths,"filename")
# execution_times = []
tp = 0
fp = 0
with detection_graph.as_default():
  with tf.Session() as sess:
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
    for group in filename_grouped:
      image_name = group.filename
      image_path = os.path.join(TEST_DIR,image_name)

      image = Image.open(image_path)
      width, height = image.size
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Run inference
      t_start = time.time()
      output_dict = sess.run(tensor_dict,
                              feed_dict={image_tensor: image_np_expanded})
      t_end = time.time()

      # Parse Preds 
      pred_bboxes = parse_predictions(output_dict,width, height)     
      # Parse Gts
      groundtruth_bboxes = parse_groundtruths(group)
      # Cal Confusion Matrix
      matrix = cal_cfs_matrix(pred_bboxes,groundtruth_bboxes)
      
      for row_matrix in matrix:
        row_matrix = np.array(row_matrix)
        if np.sum(row_matrix) > 0:
          tp += 1 
      
      for ii in range(matrix.shape[1]):
        col_matrix = matrix[:,ii]
        if np.sum(col_matrix) == 0:
          fp += 1 

      print(matrix)
      print("TP {} FP {} ".format(tp,fp))
      # execution_times.append(t_end-t_start)
      out_path = os.path.join("cache",image_name)
      # cv2.imwrite(out_path,_image)

# print(execution_times)
