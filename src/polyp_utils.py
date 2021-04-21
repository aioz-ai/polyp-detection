import numpy as np 
import pandas as pd
import tensorflow as tf 
from collections import namedtuple

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

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

def polyp_evaluation_metric(pred_bbox,groundtruth_bbox):
    _x = (pred_bbox[0]+pred_bbox[2])/2
    _y = (pred_bbox[1]+pred_bbox[3])/2
    # GT BBOX
    xmin,ymin,xmax,ymax = groundtruth_bbox
    if _x > xmin and _x < xmax and _y > ymin and _y < ymax:
        return 1
    else:
        return 0

def parse_predictions(output_dict, width, height):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    predictions = output_dict['detection_boxes']
    pred_scores = output_dict['detection_scores']
    idxs = np.where(pred_scores >= 0.5)[0]
    predictions = predictions[idxs]
    predictions = np.array(predictions)
    pred_bboxes = []
    for prediction in predictions:
        ymin = prediction[0] * height
        xmin = prediction[1] * width
        ymax = prediction[2] * height
        xmax = prediction[3] * width
        # Drawing
        # cv2.rectangle(_image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),color=(255,0,0),thickness=10)
        pred_bbox = np.array(([xmin,ymin,xmax,ymax]))
        pred_bboxes.append(pred_bbox)
    return pred_bboxes

def parse_groundtruths(group):
    groundtruth_bboxes = []
    for index, row in group.object.iterrows():
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        # cv2.rectangle(_image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),color=(0,255,0),thickness=9)
        groundtruth_bboxes.append([xmin,ymin,xmax,ymax])
    return groundtruth_bboxes

def cal_cfs_matrix(pred_bboxes, groundtruth_bboxes):
    matrix = []
    for groundtruth_bbox in groundtruth_bboxes:
        correct_flag = 0
        row_matrix = []
        for pred_bbox in pred_bboxes:
            rs = polyp_evaluation_metric(pred_bbox,groundtruth_bbox)
            row_matrix.append(rs)
        matrix.append(row_matrix)
    matrix = np.array(matrix)
    return matrix