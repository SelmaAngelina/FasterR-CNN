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
import cv2

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
    
""" VISUALIZE ALL FILTERS APPLIED TO THE IMAGE END"""      



#test with first image
dataset = tf.data.TFRecordDataset(example_proto)
dataset = dataset.map(parse_tfrecord).skip(1).take(1)

# =============================================================================
# for image in dataset:
#     image = tf.expand_dims(image,axis=0)
#     #get_feature_map(image)
#     get_conv_feature_maps(image)
# =============================================================================


""" GENERATE ANCHOR BOXES """
ASPECT_RATIOS = [0.5, 1.0, 2.0]
SCALES = [16,32,64]

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

            #boxes[i * num_anchors + j] = [x_min, y_min, x_max, y_max]
            boxes[i * num_anchors + j] = [y_min, x_min, y_max, x_max]
            
            
    ################## PLOTTING CENTER POINTS ON ORIGINAL IMAGE ####################
    # Reshape the image to remove the batch dimension
    image = tf.reshape(image, (800, 1360, 3))
    #Display the image
    plt.imshow(image)
    plt.scatter(center_xy[:,0], center_xy[:,1], s=1, c='r')
    plt.show()


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

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

import pandas as pd 

def compute_iou_matrix(bboxes, gt_boxes):
    num_anchors = bboxes.shape[0]
    num_gt_boxes = gt_boxes.shape[0]
    
    iou_matrix = np.zeros((num_anchors,num_gt_boxes))
    
    for i in range(num_gt_boxes):
        for j in range(num_anchors):
            iou_matrix[j][i] = compute_iou(bboxes[j], gt_boxes[i])
    
    # initialize a list to store the indices of valid anchor boxes
    valid_anchors = []
    
    # iterate over each anchor box
    for i in range(num_anchors):
        # get the coordinates of the anchor box
        y1, x1, y2, x2 = bboxes[i]
        
        # check if the anchor box lies within the image boundaries
        if x1 >= 0 and y1 >= 0 and x2 <= 1360 and y2 <= 800:
            valid_anchors.append(i)
    
    # create a dictionary to store the data
    data = {"anchor_idx": valid_anchors}
    
    # add the IoU scores for each object
    for idx in range(num_gt_boxes):
        data[f"object_{idx}_iou"] = iou_matrix[:, idx]
    
    # for each anchor box assign max IoU among all objects in the image
    data["max_iou"] = iou_matrix.max(axis=1)
    
    # for each anchor box assign ground truth having maximum IoU
    data["best_gt"] = iou_matrix.argmax(axis=1)
    
    # create a Pandas DataFrame from the data and return it
    df_iou = pd.DataFrame(data)
    
    return df_iou, iou_matrix

""" SAMPLE ANCHORS """

def sample_anchors(iou_df,iou_matrix,anchors):
    #get anchor boxes having max IoU for each ground truth box
    best_ious = df_iou.drop(['anchor_idx','max_iou','best_gt'],axis=1).max().values
    print(f"Top IoUs for each object in the image: {best_ious}")
    
    #get anchor box idx having max overlap with ground truth boxes 
    best_anchors = df_iou.drop(['anchor_idx','max_iou','best_gt'],axis=1).values.argmax(axis=0)
    print(f"Top anchor boxes index: {best_anchors}")
    
    
    #get all the anchor boxes having same IoU score
    top_anchors = np.where( iou_matrix == best_ious)[0]
    print(f"Anchor boxes with same IOU score: {top_anchors}")
    
    return top_anchors
    
def visualize_top_anchors(img,top_anchors,gt_boxes,anchor_boxes):
    img_ = np.copy(img)
    
    for i in top_anchors:
        y_min = int(anchor_boxes[i][0])
        x_min = int(anchor_boxes[i][1])
        y_max = int(anchor_boxes[i][2])
        x_max = int(anchor_boxes[i][3])
        cv2.rectangle(img_,(x_min,y_min), (x_max,y_max), color = (0,255,0), thickness = 2)
    
    for i, gt_box in enumerate(gt_boxes):
        y_min, x_min, y_max, x_max = gt_box
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cv2.rectangle(img_,(x_min,y_min), (x_max,y_max),color=(255, 0, 0), thickness = 2)
    
    plt.imshow(img_)
    plt.show()

""" SAMPLE ANCHORS END"""
        
    
       
for image,bboxes in dataset:
    image = tf.expand_dims(image,axis=0)
    #compute_centers(image, sizes, aspect_ratios, scales)
    feature_map, feature_map_size = get_conv_feature_maps(image)

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
    gt_boxes_scaled = np.array(gt_boxes_scaled)
    
    #compute_iou_matrix(boxes, gt_boxes)
    df_iou,iou_matrix = compute_iou_matrix(boxes, gt_boxes_scaled)
    
    """"
    # Compute the IoU between every anchor box and every ground truth box
    iou = compute_iou_matrix2(boxes, gt_boxes_scaled)
    
    # Plot the IoU matrix
    plot_iou_matrix(iou)
    """

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
    
    top_anchors = sample_anchors(df_iou, iou_matrix,boxes)
    visualize_top_anchors(img_data, top_anchors, gt_boxes_scaled, boxes)
    
    # import cv2

    # img_data = np.array(img_data)
    # # Draw rectangles around anchors and ground truth boxes
    # for i, anchor in enumerate(boxes):
    #     x_min, y_min, x_max, y_max = map(int, anchor)
    #     cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

    # for i, gt_box in enumerate(gt_boxes_scaled):
    #     x_min, y_min, x_max, y_max = map(int, gt_box)
    #     print("GTBOXES:")
    #     print(x_min, y_min, x_max, y_max)
    #     cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # # Display image
    # cv2.imshow('Anchors and Ground Truth Boxes', img_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    


    
    
    
   
    
    
    
    
    
   


  