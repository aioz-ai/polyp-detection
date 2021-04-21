import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import split
import label_map_util
from py_nms import non_max_suppression_fast

PATH_TO_FROZEN_GRAPH = "faster_precoco_0.pb"
PATH_TO_LABELS = "polyp.pbtxt"
TEST_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(TEST_DIR, image_name) for image_name in os.listdir(TEST_DIR)]

def load_model(path):
  with tf.gfile.GFile(path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
  return graph
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

detection_graph = load_model(PATH_TO_FROZEN_GRAPH)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

groundtruths = pd.read_csv("./polyp_test.csv")
filename_grouped = split(groundtruths,"filename")

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
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    for group in filename_grouped:
      image_name = group.filename
      image_path = os.path.join(TEST_DIR,image_name)

      image = Image.open(image_path)
      width, height = image.size
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Run inference
      output_dict = sess.run(tensor_dict,
                              feed_dict={image_tensor: image_np_expanded})
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

      pred_bboxes = output_dict['detection_boxes']
      pred_scores = output_dict['detection_scores']
      idxs = np.where(pred_scores >= 0.5)[0]
      pred_num = idxs.shape[0]
      pred_bboxes = pred_bboxes[idxs]
      
      pred_bboxes = np.array(pred_bboxes)
      final_bboxes = []
      print("PRED:")
      for pred_bbox in pred_bboxes:
        ymin = pred_bbox[0] * height
        xmin = pred_bbox[1] * width
        ymax = pred_bbox[2] * height
        xmax = pred_bbox[3] * width
        final_bboxes.append([xmin,ymin,xmax,ymax])
      gt_num = 0
      print("GT:")
      for index, row in group.object.iterrows():
        gt_num += 1
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        final_bboxes.append([xmin,ymin,xmax,ymax])
      final_bboxes = np.array(final_bboxes).astype(np.float32)
      print(final_bboxes)
      final_bboxes = non_max_suppression_fast(final_bboxes,0.5)
      tp = pred_num + gt_num - final_bboxes.shape[0]
      fp = max(pred_num - tp,0)
      fn = max(gt_num - tp,0)
      sum_tp += tp
      sum_fp += fp
      sum_fn += fn
      print("TP {} FP {} FN {}".format(sum_tp,sum_fp,sum_fn))