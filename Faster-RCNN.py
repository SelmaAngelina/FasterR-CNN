"""Region Proposal Network"""

""" BASE LAYERS - CONVOLUTIONAL BACKBONE NETWORK FOR OUTPUT FEATURE MAP"""
# implementation of VGG-16 backbone for use as a feature extractor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import Model

# this is the file path to the TFRecord
example_proto = "/Users/selmamusledin/Desktop/CV - Traffic Sign Detection/TFRecords/train.record"

from PIL import Image
from io import BytesIO

TARGET_SIZE = (800,1360) #VGG16 requires images to be resized in a square-shaped input size

def parse_tfrecord(example_proto):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    example = tf.io.parse_single_example(example_proto, feature_description) #parse a single example from a serialized string tensor

    width = tf.cast(example['image/width'], tf.float32)
    height = tf.cast(example['image/height'], tf.float32)
    
    #decode the image into a uint8 tensor (ranging between 0 and 255)
    image = tf.image.decode_jpeg(example['image/encoded'],channels=3)
    
    #reshape the image to the desired shape of VGG16
    image = tf.reshape(image, (height,width,3))
    
    #resize the image to match the input size of the VGG16 network
    # image = tf.image.resize(image, (800,1360)) 
    
    #preprocess the image to match VGG16 network's expectations
    #image = tf.keras.applications.vgg16.preprocess_input(image) #scales pixel values and performs mean substraction to match the preprocessing used during training of the VGG16 network 
        
    #extract bbox coordinates
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    
    #scale bbox coordinates to match the resized image
    # xmin = xmin * TARGET_SIZE[0]/width
    # xmax = xmax * TARGET_SIZE[0]/width
    # ymin = ymin * TARGET_SIZE[1] / height
    # ymax = ymax * TARGET_SIZE[1] / height
    
    #concatenate the scaled bbox coordinates
    bboxes = tf.stack([ymin,xmin,ymax,xmax],axis=-1)
    
    return image,bboxes
    

    
from tensorflow.keras.preprocessing import image
def get_feature_map(image):
    #create the VGG16 model with the pre-trained weights
    vgg_model = tf.keras.applications.VGG16(weights='imagenet',include_top=False)
    
    #get the output of the last convolutional layer 
    last_conv_layer = vgg_model.get_layer('block5_conv3').output
    
    #create a new model that outputs the feature maps from the last convolutional layer
    feature_extractor = tf.keras.Model(inputs=vgg_model.input,outputs = last_conv_layer)
    
    #generate the feature maps for the preprocessed image
    image_feature_map = feature_extractor(image)
    
    """VISUALIZE THE FEATURE MAP"""
    #get the feature map from the VGG16 model
    feature_map = feature_extractor.predict(image)
    print('Feature map shape is: ' ,feature_map.shape)
    
    #get the dimensions of the feature map
    feature_height, feature_width, feature_depth = feature_map.shape[1:]
    
     #convert the map tensor to an image
    img_array = feature_map[0,:,:,0] #visualize the first channel of the feature map
    img_array = (img_array - img_array.min()) / ((img_array.max() - img_array.min())) # normalize the values between 0 and 1
    img_array = img_to_array(img_array)
     
    #display the original image and the feature map side by side
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(image[0]) #first image in the batch
    axs[0].set_title('Original image')
    axs[1].imshow(img_array)
    axs[1].set_title('Feature Map')
    plt.show()
    
    return feature_map, (feature_height,feature_width,feature_depth)





""" VISUALIZE ALL FILTERS APPLIED TO THE IMAGE """
#create a new model that takes the original image as input and outputs the feature map at each convolutional layer
def get_conv_feature_maps(image):
    vgg_model = tf.keras.applications.VGG16(weights='imagenet',include_top=False)
    
    #define a model that outputs the feature map after each convolutional layer
    outputs = [layer.output for layer in vgg_model.layers if 'conv' in layer.name]
    feature_extractor = tf.keras.models.Model(inputs=vgg_model.input,outputs=outputs)
    
    #image is already preprocessed when parsing the TFRecord
    feature_maps = feature_extractor.predict(image)
    
    feature_maps_sizes = [fm.shape[1:3] for fm in feature_maps]
    print("Feature Map Sizes are: ", feature_maps_sizes)
    
    #visualize the feature maps
    for i,fm in enumerate(feature_maps):
        fig,axs = plt.subplots(1,2,figsize=(10,5))
        axs[0].imshow(image[0])
        axs[0].set_title('Original image')
        axs[1].imshow(fm[0,:,:,0])
        axs[1].set_title('Layer {} Feature Maps'.format(i+1))
        plt.show()
    
    return feature_maps, feature_maps_sizes
    
        

#test with first image
dataset = tf.data.TFRecordDataset(example_proto)
dataset = dataset.map(parse_tfrecord).skip(3).take(1)

# =============================================================================
# for image in dataset:
#     image = tf.expand_dims(image,axis=0)
#     #get_feature_map(image)
#     get_conv_feature_maps(image)
# =============================================================================


""" GENERATE ANCHOR BOXES """

from tqdm import tqdm

def generate_anchors(image, feature_map_size, aspect_ratios, scales, sizes):

    """ return: array of anchor boxes """
    #compute anchor box dimensions for all ratios and scales at all levels of the feature pyramid
    anchor_dims_all = []
    for size in tqdm(sizes,desc='Generating anchor boxes'):
        anchor_dims=[]
        for ratio in aspect_ratios:
            height = np.sqrt(size/ratio)
            width = size/height
            dims = np.stack([width,height],axis=-1)
            for scale in scales:
                anchor_dims.append(scale * dims)
        anchor_dims_all.append(np.stack(anchor_dims,axis=-2))
    
    #compute the anchor box centers for each feature level
    num_levels = len(anchor_dims_all)
    print("Num levels: ", num_levels)
    anchors_all = []
    
    for level in range(num_levels):
        stride = 2 ** (level + 2)
        
        #getting feature height and feature width for each level of the feature pyramid
        feature_height, feature_width = feature_map_size[level]
        
        #get center points
        rx = np.arange(feature_width,dtype=np.float32) + 0.5
        ry = np.arange(feature_height,dtype=np.float32) + 0.5
        
        centers = np.stack(np.meshgrid(rx,ry),axis=-1) * stride
        
        #reshape anchor dimensions and centers for broadcasting
        anchor_dims = anchor_dims_all[level] #get all anchor_dims for each level of the feature pyramid
        num_anchors = anchor_dims.shape[0] #get the number of anchors for the first spatial location of the feature pyramid at a given level
        
        anchors = np.zeros((feature_height,feature_width,num_anchors,4))
        anchors[:,:,:,:2] = centers.reshape((feature_height,feature_width,1,2))
        anchors[:,:,:,2:] = anchor_dims.reshape((1,1,num_anchors,2))
        anchors_all.append(anchors.reshape((-1,4)))
        
    #Convert anchors from center-offset format to corner format
    if len(image.shape) == 4:
        image_height, image_width, _ = image.shape[1:]
    else:
        image_height, image_width, _ = image.shape

    grid_x,grid_y = np.meshgrid(np.arange(feature_width),np.arange(feature_height))
    anchor_widths, anchor_heights = anchors_all[0][:, 2], anchors_all[0][:, 3]
    anchor_centers_x, anchor_centers_y = anchors_all[0][:, 0], anchors_all[0][:, 1]
    x_min = anchor_centers_x - 0.5 * anchor_widths
    y_min = anchor_centers_y - 0.5 * anchor_heights
    x_max = anchor_centers_x + 0.5 * anchor_widths
    y_max = anchor_centers_y + 0.5 * anchor_heights
    anchors_all[0] = np.stack([x_min, y_min, x_max, y_max], axis=-1)
    
    # Clip anchors to image boundaries
    anchors_all = np.concatenate(anchors_all, axis=0)
    anchors_all[:, [0, 2]] = np.clip(anchors_all[:, [0, 2]], 0, image_width - 1)
    anchors_all[:, [1, 3]] = np.clip(anchors_all[:, [1, 3]], 0, image_height - 1)

    return anchors_all #generates anchor boxes for a given input image and feature map size



def visualize_anchors_on_feature_map(feature_map, anchors, image_height, image_width):
    """
    Visualize the anchor boxes on top of the feature map.
    """
    num_anchors = anchors.shape[0]
    colors = ['b','c','g','k','m','r','w','y','b']
    for i in tqdm(range(num_anchors),desc="Plotting"):
        x1,y1,x2,y2 = anchors[i]
        x1 = max(0,x1)
        y1 = max(0, y1)
        x2 = min(image_width - 1, x2)
        y2 = min(image_height - 1, y2)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colors[np.random.randint(0,9)], linewidth=1))
               
    #convert the map tensor to an image
    img_array = feature_map[0,:,:,0] #visualize the first channel of the feature map
    img_array = (img_array - img_array.min()) / ((img_array.max() - img_array.min())) # normalize the values between 0 and 1
    img_array = img_to_array(img_array)
    plt.imshow(img_array)
    plt.show()
    
def rgb_to_bgr(rgb_color):
    # Convert an RGB color to a BGR color
    return tuple(np.flip(rgb_color))


def resize_anchors(anchors, target_size):
    """
    Resizes a list of anchor boxes based on a target size.

    Args:
        anchors: A list of anchor boxes.
        target_size: A tuple of (height, width) representing the target size.

    Returns:
        A list of resized anchor boxes.
    """
    resized_anchors = []
    for anchor in anchors:
        y_min, x_min, y_max, x_max = anchor
        h_scale = target_size[0] / (y_max - y_min)
        w_scale = target_size[1] / (x_max - x_min)
        y_min *= h_scale
        x_min *= w_scale
        y_max *= h_scale
        x_max *= w_scale
        resized_anchors.append([y_min, x_min, y_max, x_max])
        
    return resized_anchors


import cv2     
def visualize_anchors_on_image(image, anchors, anchor_colors=None, box_thickness=2):
    """
    Draws the bounding boxes of the anchors on the original image.

    Args:
        image: The original image.
        anchors: A list of anchor boxes.
        anchor_colors: A list of colors for the anchor boxes. If None, all anchor boxes will be drawn in red.
        box_thickness: The thickness of the box lines.

    Returns:
        The image with the anchor boxes drawn on it.
    """
    # Rescale the anchors to the original image size
    image_height, image_width, image_depth = image.shape
    resized_anchors = resize_anchors(anchors, (image_height,image_width))

    # Define the default anchor color
    if anchor_colors is None:
        anchor_colors = [(0, 0, 255)] * len(resized_anchors)
    else:
        # Convert the RGB anchor colors to BGR colors
        anchor_colors = [rgb_to_bgr(color) for color in anchor_colors]

    # Draw the anchor boxes on the image
    image_with_boxes = image.copy()
    for anchor, color in zip(resized_anchors, anchor_colors):
        ymin, xmin, ymax, xmax = anchor
        xmin = int(xmin * image_width)
        xmax = int(xmax * image_width)
        ymin = int(ymin * image_height)
        ymax = int(ymax * image_height)
        image_with_boxes = cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color=color, thickness=box_thickness)

    return image_with_boxes

""" NEW """
ASPECT_RATIOS = [0.5, 1.0, 2.0]
SCALES = [16,32,64]

# def generate_anchor_boxes(input_image, feature_map_size, aspect_ratios, scales):
#     #compute stride -> same for height and width
#     stride = input_image.shape[1] // feature_map_size[0]
    
#     #generate anchor box centers
#     x_centers = tf.range(feature_map_size[1], dtype = tf.float32) * stride + stride/2
#     y_centers = tf.range(feature_map_size[0], dtype = tf.float32) * stride + stride/2
    
#     x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
    
#     x_centers = tf.reshape(x_centers,(-1,))
#     y_centers = tf.reshape(y_centers,(-1,))
    
#     #generate anchor box coordinates
#     num_anchors = len(scales) * len(aspect_ratios)
#     boxes = tf.zeros((num_anchors * feature_map_size[0] * feature_map_size[1], 4))
#     idx = 0
    
#     for scale in scales:
#         for aspect_ratio in aspect_ratios:
#             width = scale * tf.sqrt(aspect_ratio)
#             height = scale / tf.sqrt(aspect_ratio)

import numpy as np

def generate_anchor_boxes(image,feature_map_shape, input_shape, scales=[32,64,128], aspect_ratios=[0.5, 1.0, 2.0]):
    # Compute the stride based on the feature map shape and input shape
    stride_w = input_shape[1] // feature_map_shape[1]
    print("Stride_w is: ", stride_w)
    
    stride_h = input_shape[0] // feature_map_shape[0]
    print("Stride_h is: ", stride_w)

    # Compute the coordinates of the center pixel for each feature map cell
    center_x = np.arange(stride_w // 2, input_shape[1], stride_w)
    center_y = np.arange(stride_h // 2, input_shape[0], stride_h)
    center_x, center_y = np.meshgrid(center_x, center_y)

    # Reshape the center coordinates to be a list of (x, y) tuples
    center_x = center_x.ravel()
    center_y = center_y.ravel()
    center_xy = np.stack((center_x, center_y), axis=1)
    print("center_xy is: ", center_xy)

    # Compute the width and height of the anchor boxes for each scale and aspect ratio
    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios)
    num_anchors = len(scales) * len(aspect_ratios)
    widths = (scales.reshape(-1, 1) * np.sqrt(aspect_ratios).reshape(1, -1)).reshape(-1)
    heights = (scales.reshape(-1, 1) / np.sqrt(aspect_ratios).reshape(1, -1)).reshape(-1)

    # Compute the bounding box coordinates for each anchor box and center
    boxes = np.zeros((len(center_xy) * num_anchors, 4))
    for i, (x, y) in enumerate(center_xy):
        for j, (w, h) in enumerate(zip(widths, heights)):
            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2

            # Project the anchor box coordinates onto the input image
            x_min = max(0, min(x_min, input_shape[1] - 1))
            y_min = max(0, min(y_min, input_shape[0] - 1))
            x_max = max(0, min(x_max, input_shape[1] - 1))
            y_max = max(0, min(y_max, input_shape[0] - 1))

            boxes[i * num_anchors + j] = [x_min, y_min, x_max, y_max]
            
            
    ################## PLOTTING CENTER POINTS ON ORIGINAL IMAGE ####################
    # Reshape the image to remove the batch dimension
    #image = tf.reshape(image, (800, 1360, 3))
    # Display the image
    #plt.imshow(image)
    #plt.scatter(center_xy[:,0], center_xy[:,1], s=1, c='r')
    #plt.show()


    return boxes



def visualize_anchor_boxes(image, boxes, image_size):
    """
    Displays the anchor boxes on the input image.
    """
    image_with_boxes = np.copy(image)

    # Convert the boxes to the format expected by cv2.rectangle
    boxes = np.round(boxes * image_size).astype(np.int32)

    # Define a unique color for each anchor box
    num_boxes = len(boxes)
    colors = [(np.random.rand(3) * 255).astype(int) for _ in range(num_boxes)]

    # Draw a rectangle for each anchor box
    for i, box in enumerate(boxes):
        color = colors[i]
        x1, y1, x2, y2 = box
        image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color.astype(np.float32).tolist(), 2)

    # Display the image
    plt.imshow(image_with_boxes)
    plt.show()


""" I have generated anchor box coordinates 
        -> 1.visualize center points on original image
        -> 2.visualize all anchors for one center point on the original image
"""
from matplotlib import patches
def plot_anchor_boxes(anchor_boxes, image):
    """
    Plots anchor boxes on an image.
    
    Args:
    anchor_boxes: numpy array of shape (num_boxes, 4) containing the anchor box coordinates.
    image: numpy array of shape (height, width, channels) containing the input image.
    """
    colors = ['r', 'g', 'b', 'y', 'm']
    num_colors = len(colors)

    # Randomly assign a color to each anchor box
    color_indices = np.random.randint(num_colors, size=len(anchor_boxes))

    # Plot the anchor boxes on the input image
    fig, ax = plt.subplots()
    image = tf.reshape(image, (800, 1360, 3))
    ax.imshow(image)

    for i, box in enumerate(anchor_boxes):
        color_index = color_indices[i]
        color = colors[color_index]

        # Plot the anchor box
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()



"""
    IOU
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
    
    #check if they at least overlap a little
    if(x_min < x_max and y_min < y_max):
        w_overlap = (x_max - x_min)
        h_overlap = (y_max - y_min)
        area_overlap = w_overlap * h_overlap
    else:
        #in case there is no overlap
        return 0
    
    #-----computing union-----
    width_box1 = (box1[2] - box1[0])
    height_box1 = (box1[3] - box1[1])
    
    width_box2 = (box2[2] - box2[0])
    height_box2 = (box2[3] - box2[1])
    
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    area_union_overlap = area_box1 + area_box2
    
    area_union = area_union_overlap - area_overlap
    
    iou = area_overlap / area_union
    
    return iou

import pandas as pd 
def compute_iou_matrix(bboxes, gt_boxes):
    num_anchors = bboxes.shape[0]
    num_gt_boxes = gt_boxes.shape[0]
    
    iou_matrix = np.zeros((num_anchors,num_gt_boxes))
    
    # for i in range(num_anchors):
    #     for j in range(num_gt_boxes):
    #         iou_matrix[i,j] = compute_iou(bboxes[i], gt_boxes[j])
    
    for i, gt_box in enumerate(gt_boxes):
        for j, anchor in enumerate(bboxes):
            iou_matrix[j][i] = compute_iou(anchor, gt_box)

    
    anchor_idx_list = np.where((bboxes[:, 0] >= 0) &(bboxes[:, 1] >= 0) & (bboxes[:, 2] <= 1360) & (bboxes[:, 3] <= 800))[0]
    data = {"anchor_idx": anchor_idx_list}
    
    data.update({f"object_{idx}_iou":iou_matrix[:, idx] for idx in range(num_gt_boxes)})
    
    # for each anchor box assign max IOU among all objects in the image
    data["max_iou"] = iou_matrix.max(axis= 1)

    # for each anchorbox assign ground truth having maximum IOU
    data["best_gt"] = iou_matrix.argmax(axis= 1)

    df_iou = pd.DataFrame(data)
    
    return df_iou

       
for image,bboxes in dataset:
    image = tf.expand_dims(image,axis=0)
    #compute_centers(image, sizes, aspect_ratios, scales)
    #feature_map, feature_map_size = get_conv_feature_maps(image)

    #################### NEW #########################
    
    """
    Get feature map + feature map size
    """
    feature_map, feature_map_size = get_feature_map(image)
    boxes = generate_anchor_boxes(image,feature_map_size, image.shape[1:3])
    
    #get ground truth bouding boxes
    gt_boxes = bboxes
    
    """ VISUALZIE """
    #img = np.squeeze(image)
    #img_size = img.shape[2]
    #visualize_anchor_boxes(img, boxes,img_size)
    
    # Select a pixel in the feature map to plot anchor boxes for
    pixel_index = 2500#2167
    pixel_anchor_boxes = boxes[pixel_index*9:(pixel_index+1)*9]

    # Plot the anchor boxes for the selected pixel on the input image
    plot_anchor_boxes(pixel_anchor_boxes, image)
   
    # Scale the bounding boxes to match the original image size
    gt_boxes_scaled = gt_boxes * [800, 1360, 800, 1360]
    
    #compute_iou_matrix(boxes, gt_boxes)
    df_iou = compute_iou_matrix(boxes, gt_boxes_scaled)
    
    
    #plot gt
    fig, ax = plt.subplots()
    img_data = image[0,...]
    
    # Plot the image and bounding boxes
    ax.imshow(img_data)

    for box in gt_boxes_scaled:
        ymin, xmin, ymax, xmax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    
    
    
   
    
    
    
    
    
   


  