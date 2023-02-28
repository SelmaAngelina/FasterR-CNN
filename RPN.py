""" GENERATE ANCHOR BOXES """
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import read_data 
import matplotlib.pyplot as plt
import PIL

ASPECT_RATIOS = [1,0.5,2]
SCALES = [128,256, 512]

######## Generate Anchor Boxes with different shapes centered on each pixel of the image ########

def min_max_to_center(xmin,ymin,xmax,ymax):
    height = ymax-ymin
    width = xmax-xmin
    center_x = xmin + width/2
    center_y = ymin + height/2
    
    return width, height, center_x,center_y

def adjust_deltas(anchor_width,anchor_height,anchor_center_x,anchor_center_y,dx,dy,dw,dh):
    """ Adjust the anchor box with the predicted delta """
    center_x = dx * anchor_width + anchor_center_x
    center_y = dy + anchor_height + anchor_center_y
    
    width = np.exp(dw) * anchor_width
    height = np.exp(dy) * anchor_height
    
    return center_x,center_y,width,height

def deltas(base_center_x,base_center_y,base_width,base_height, inside_anchor_width, inside_anchor_height, inside_anchor_center_x,inside_anchor_center_y):
    """ Get offset of anchor to ground truth """
    dx = (base_center_x - inside_anchor_center_x) / inside_anchor_width
    dy = (base_center_y- inside_anchor_center_y) / inside_anchor_height
    
    dw = np.log(base_width / inside_anchor_width)
    dh = np.log(base_height / inside_anchor_height)
    
    return dx,dy,dw,dh






""" RPN - IMPLEMENTATION """
"""
    ->for training, we take all the anchors and put them into two different categories
        -those that overlap a ground-truth object with IoU > 0.5 (foreground)
        -those that don't overlap any ground-truth object or IoU < 0.1 (background)
        
    ->randomly sample those anchors to form a mini-batch of size 256 (try to maintain a balanced ratio between background and foreground)
"""

def compute_iou(box1,box2):
    #compute coordinates of intersection rectangle
    x_min = max(box1[0],box2[0])
    y_min = max(box1[1],box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    #compute area of intersection
    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
    
    #computea area of union
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area
    
    #compute IoU
    iou = intersection_area/ union_area
    
    return iou


def compute_iou_matrix(bboxes, gt_boxes):
    num_anchors = bboxes.shape[0]
    num_gt_boxes = bboxes.shape[0]
    
    iou_matrix = np.zeros((num_anchors,num_gt_boxes))
    
    for i in range(num_anchors):
        for j in range(num_gt_boxes):
            iou_matrix[i,j] = compute_iou(bboxes[i], gt_boxes[j])
    
    return iou_matrix




















###############################################################################
import cv2
""" Create and visualize annotations """
def get_example_and_plot(example_proto, height, width):
    #read the TFRecord and extract all the relevant data
    data = tf.data.TFRecordDataset(example_proto)
    for record in data.skip(1).take(1):
        example = tf.train.Example() #a {"string": tf.train.Feature} mapping
        example.ParseFromString(record.numpy())
        
        #decode the image from its serialized format
        image_string = example.features.feature['image/encoded'].bytes_list.value[0]
        image = tf.image.decode_jpeg(image_string)
        image = np.array(image)
        
        #plot the image
        plt.imshow(image)
        
        #extract bounding box coordinates
        xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
        ymin = example.features.feature['image/object/bbox/ymin'].float_list.value
        xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
        ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
        
        #convert bounding box coordinates to image dimensions
        xmin = [x * width for x in xmin]
        ymin = [y * height for y in ymin]
        xmax = [x * width for x in xmax]
        ymax = [y * height for y in ymax]
        
        #overlay bounding boxes on the image
        for xmin_i, ymin_i, xmax_i, ymax_i in zip(xmin, ymin, xmax, ymax):
             plt.gca().add_patch(plt.Rectangle((xmin_i, ymin_i), xmax_i - xmin_i, ymax_i - ymin_i, linewidth=1, edgecolor='r', facecolor='None'))


example_proto = "/Users/selmamusledin/Desktop/CV - Traffic Sign Detection/TFRecords/train.record"
#get_example_and_plot(example_proto,800,1360)

