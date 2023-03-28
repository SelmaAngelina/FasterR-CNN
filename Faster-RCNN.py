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
import pandas as pd

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
    
    label = tf.sparse.to_dense(example['image/object/class/label'])
    labels = tf.cast(label, tf.int64)
    
    # Reshape the labels to be a 1D tensor
    labels = tf.reshape(labels, [-1])
    
    
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
    
    
    return image,bboxes,labels
    

    
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




""" GENERATE ANCHOR BOXES """

def generate_anchor_boxes(image,feature_map_shape, input_shape, scales=[16,32,64], aspect_ratios=[0.75,1.0,2.0]):
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
    #image = tf.reshape(image, (800, 1360, 3))
    #Display the image
    #plt.imshow(image)
    #plt.scatter(center_xy[:,0], center_xy[:,1], s=1, c='r')
    #plt.show()


    return boxes

"""NEW TRY"""

ASPECT_RATIOS = [0.5, 1.0, 1.5]
SCALES = [3, 4, 5]

def generate_center_points(image, feature_map_shape, input_shape):
    stride_w = input_shape[1] // feature_map_shape[1]
    print("Stride_w is: ", stride_w)
    print("feature_map_shape[1] is " , feature_map_shape[1])
    
    stride_h = input_shape[0] // feature_map_shape[0]
    print("Stride_h is: ", stride_w)
    print("feature_map_shape[0] is " , feature_map_shape[0])
    
    # center (xy) coordinates of anchor location on the image
    x_center = np.arange(8,input_shape[1], stride_w)
    y_center = np.arange(8, input_shape[0], stride_h)
    
    #generate all ordered pairs of (xy)
    center_list = np.array(np.meshgrid(x_center, y_center,  sparse=False, indexing='xy')).T.reshape(-1,2)
    
    # Reshape the image to remove the batch dimension
    image = tf.reshape(image, (800, 1360, 3))
    #visualize anchor positions on the image
    img_ = np.copy(image)
    
    plt.figure(figsize=(9,6))
    n_anchor_pos = feature_map_shape[0] * feature_map_shape[1]
    for i in range(n_anchor_pos):
        cv2.circle(img_, (int(center_list[i][0]), int(center_list[i][1])), radius=1, color=(255, 0, 0), thickness=5) 
    plt.imshow(img_)
    plt.show()
    
    return center_list
        
    
def generate_anchor_boxes2(gt_boxes,center_list, image,feature_map_shape, input_shape):
    al=[]
    #total possible anchors
    n_anchors = feature_map_shape[0] * feature_map_shape[1] * len(ASPECT_RATIOS) * len(SCALES)
    
    #number of objects in the image
    n_object = len(gt_boxes)
    
    anchor_list = np.zeros(shape = (n_anchors,4))
    
    count = 0
    
    x_stride = input_shape[1] // feature_map_shape[1]
    y_stride = input_shape[0] // feature_map_shape[0]
    
    for center in center_list:
        center_x, center_y = center[0], center[1]
        # for each ratio
        for ratio in ASPECT_RATIOS:
            # for each scale
            for scale in SCALES:
                # compute height and width and scale them by constant factor
                h = pow(pow(scale, 2)/ ratio, 0.5)
                w = h * ratio

                # as h and w would be really small, we will scale them with some constant (in our case, stride width and height)
                h *= x_stride
                w *= y_stride


                # * at this point we have height and width of anchor and centers of anchor locations
                # putting anchor 9 boxes at each anchor locations
                anchor_xmin = center_x - 0.5 * w
                anchor_ymin = center_y - 0.5 * h
                anchor_xmax = center_x + 0.5 * w
                anchor_ymax = center_y + 0.5 * h
                al.append([center_x, center_y, w, h])
                # append the anchor box to anchor list
                anchor_list[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
                count += 1
    
    return anchor_list


def visualize_anchor_at_center_location(center_list,image, anchor_list, gt_boxes):
    # visualize anchor boxs at center anchor location
    img_ = np.copy(tf.reshape(image, (800, 1360, 3)))
    # mid anchor center = 4250/2 = 2125
    for i in range(19125, 19134):  
        x_min = int(anchor_list[i][0])
        y_min = int(anchor_list[i][1])
        x_max = int(anchor_list[i][2])
        y_max = int(anchor_list[i][3])
        cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 
        
    for i, bbox in enumerate(gt_boxes):
        ymin, xmin, ymax, xmax = bbox.astype(int)
        cv2.rectangle(img_, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)
            
    cv2.circle(img_, (int(center_list[312][0]), int(center_list[312][1])), radius=1, color=(0, 0, 255), thickness=15) 
            
    plt.imshow(img_)
    plt.show()

def IOU(box1, box2):
    """
    Compute overlap (IOU) between box1 and box2
    """
    
    # ------calculate coordinate of overlapping region------
    # take max of x1 and y1 out of both boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    
    # take min of x2 and y2 out of both boxes
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # check if they atleast overlap a little
    if (x1 < x2 and y1 < y2):
        # ------area of overlapping region------
        width_overlap = (x2 - x1)
        height_overlap = (y2 - y1)
        area_overlap = width_overlap * height_overlap
    else:
        # there is no overlap
        return 0
    
    # ------computing union------
    # sum of area of both the boxes - area_overlap
    
    # height and width of both boxes
    width_box1 = (box1[2] - box1[0])
    height_box1 = (box1[3] - box1[1])
    
    width_box2 = (box2[2] - box2[0])
    height_box2 = (box2[3] - box2[1])
    
    # area of box1 and box2
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # union (including 2 * overlap area (double count))
    area_union_overlap = area_box1 + area_box2
    
    # union
    area_union = area_union_overlap - area_overlap
    
    # compute IOU
    iou = area_overlap/ area_union
    
    return iou

def compute_iou_matrix3(anchor_list, gt_boxes):
    num_anchors = anchor_list.shape[0]
    print("Num_anchors is: " ,num_anchors)
    num_gt_boxes = gt_boxes.shape[0]
    print("Num gt_boxes is: " ,num_gt_boxes)

    iou_matrix = np.zeros((num_anchors,num_gt_boxes))

    # Transpose gt_boxes array before computing IoU
    for i in range(num_gt_boxes):
        (ymin,xmin,ymax,xmax) = gt_boxes[i]
        for j in range(num_anchors): 
            iou_matrix[j][i] = IOU(anchor_list[j], (xmin,ymin,xmax,ymax))

           
    # initialize a list to store the indices of valid anchor boxes
    valid_anchors = []
    
    # iterate over each anchor box
    for i in range(num_anchors):
        # get the coordinates of the anchor box
        x1,y1,x2,y2 = anchor_list[i]
        
        # check if the anchor box lies within the image boundaries
        if x1 >= 0 and y1 >= 0 and x2 <= 1360 and y2 <= 800:
            valid_anchors.append(i)
    print(len(valid_anchors))
    
    # create a dictionary to store the data
    data = {"anchor_idx": valid_anchors}
    
    # add the IoU scores for each object
    for idx in range(num_gt_boxes):
        data[f"object_{idx}_iou"] = iou_matrix[valid_anchors, idx]

    
    # for each anchor box assign max IoU among all objects in the image
    data["max_iou"] = iou_matrix[valid_anchors].max(axis=1)
    
    # for each anchor box assign ground truth having maximum IoU
    data["best_gt"] = iou_matrix[valid_anchors].argmax(axis=1)
    
    for key in data:
        print(key, "length:", len(data[key]))
        
    valid_anchor_list=[]
    
    for anchor_box_index in valid_anchors:
        valid_anchor_list.append(anchor_list[anchor_box_index])

    
    # create a Pandas DataFrame from the data and return it
    df_iou = pd.DataFrame(data)
    
    return df_iou, iou_matrix, np.array(valid_anchor_list)


"""END NEW TRY"""
    
    


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


""" SAMPLE ANCHORS """

def sample_anchors(df_iou,iou_matrix,anchors):
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
        x_min = int(anchor_boxes[i][0])
        y_min = int(anchor_boxes[i][1])
        x_max = int(anchor_boxes[i][2])
        y_max = int(anchor_boxes[i][3])
        
        print("Object: " ,x_min,y_min,x_max,y_max)
        cv2.rectangle(img_,(x_min,y_min), (x_max,y_max), color = (0,255,0), thickness = 2)
    
    for i, gt_box in enumerate(gt_boxes):
        y_min, x_min, y_max, x_max = gt_box
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cv2.rectangle(img_,(x_min,y_min), (x_max,y_max),color=(255, 0, 0), thickness = 2)
    
    plt.imshow(img_)
    plt.show()

def label_anchors(df_iou, top_anchors):
    # create dummy column
    label_column = np.full(df_iou.shape[0], -1, dtype=np.int)

    # label top anchor boxes as 1 # contains object
    label_column[top_anchors] = 1

    # label anchor boxes having IOU > 0.5 as 1 (foreground)
    label_column[np.where(df_iou.max_iou.values >= 0.5)[0]] = 1

    # label anchor boxes having IOU < 0.2 as 0 (background)
    label_column[np.where(df_iou.max_iou.values < 0.2)[0]] = 0

    # add column to the iou dataframe
    df_iou["label"] = label_column
    
    return df_iou


def balance_anchors_RPN(df_iou, no_samples = 256, negative_ratio = 0.5):
    no_foreground = int((1-negative_ratio) * no_samples)
    no_backgrond = int(negative_ratio * no_samples)
    
    foreground_idx = df_iou[df_iou.label == 1].index.values
    background_idx = df_iou[df_iou.label == 0].index.values
    
    #check for excessive positive samples
    if len(foreground_idx) > no_foreground:
        df_iou.loc[foreground_idx[no_foreground:], "label"] = -1

    
    #sample background examples if not enough positive examples
    if len(foreground_idx) < no_foreground:
        no_backgrond += no_foreground - len(foreground_idx)
        
    #check if we have exessive background samples
    if len(background_idx) > no_backgrond:
        df_iou.loc[background_idx[no_backgrond:], "label"] = -1
    
    return df_iou
    

""" SAMPLE ANCHORS END"""


"""
COMPUTING ANCHOR OFFSETS WITH CORRESPONDING GT BOXES
"""

def to_VOC(width, height, center_x, center_y):
    """
    Convert center coordinate format to min max coordinateformat
    """
    x_min = center_x - 0.5 * width
    y_min = center_y - 0.5 * height
    x_max = center_x + 0.5 * width
    y_max = center_y + 0.5 * height
    return x_min, y_min, x_max, y_max

def to_center(xmin_list, ymin_list, xmax_list, ymax_list):
    """
    Convert min max coordinate format to x_center, y_center, height and width format
    """
    height = ymax_list - ymin_list
    width = xmax_list - xmin_list
    
    center_x = xmin_list + 0.5 * width
    center_y = ymin_list + 0.5 * height
    
    return width, height, center_x, center_y

def adjust_deltas(anchor_width, anchor_height, anchor_center_x, anchor_center_y, dx, dy, dw, dh):
    """
    Adjust the anchor box with predicted offset
    """
    center_x = dx * anchor_width + anchor_center_x 
    center_y = dy *  anchor_height + anchor_center_y
    width = np.exp(dw) * anchor_width
    height = np.exp(dh) * anchor_height
    
    return width, height, center_x, center_y

def compute_deltas(base_center_x, base_center_y, base_width, base_height, inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y):
    """
    computing offset of achor box to the groud truth box
    """
    dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
    dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
    dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
    dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height
    return dx, dy, dw, dh
    

def convert_corners_to_center(df_iou,anchor_list,inside_anchor_list, gt_boxes):
    #anchor and gt boxes must be converted to center coordinates (xmin,ymin,xmax,ymax)
    inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y =  to_center(
        inside_anchor_list[:, 0], 
        inside_anchor_list[:, 1],
        inside_anchor_list[:, 2],
        inside_anchor_list[:, 3])
    
    #for each gt box that corresponds to each anchor box coordinate, convert coordinate format
    gt_coords=[]
    
    gt_coordinates = []
    for idx in df_iou.best_gt:
        gt_coordinates.append(gt_boxes[idx])
    gt_coordinates = np.array(gt_coordinates)
    
    base_width, base_height, base_center_x, base_center_y =  to_center(
        gt_coordinates[:, 1], 
        gt_coordinates[:, 0],
        gt_coordinates[:, 3],
        gt_coordinates[:, 2])
    
    # the code below prevents from "exp overflow"
    eps = np.finfo(inside_anchor_width.dtype).eps
    inside_anchor_height = np.maximum(inside_anchor_height, eps)
    inside_anchor_width = np.maximum(inside_anchor_width, eps)

     # computing offset given by above expression
    dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
    dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
    dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
    dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height
    
    #add offsets to df
    df_iou["dx"] = dx
    df_iou["dy"] = dy
    df_iou["dw"] = dw
    df_iou["dh"] = dh
    
    #label all possible anchors
    label_list = np.empty(len(ASPECT_RATIOS) * len(SCALES) * 50 * 85, dtype = np.float32)
    label_list.fill(-1)
    label_list[df_iou.anchor_idx.values] = df_iou.label.values
    label_list = np.expand_dims(label_list, 0)
    label_list = np.expand_dims(label_list, -1)
    
    #offsets of all possible anchors
    offset_list = np.empty(shape= anchor_list.shape, dtype= np.float32)
    offset_list.fill(0)
    offset_list[df_iou.anchor_idx.values] = df_iou[["dx", "dy", "dw", "dh"]].values
    offset_list = np.expand_dims(offset_list, 0)
    
    # combine deltas and objectiveness score in one array
    offset_list_label_list = np.column_stack((offset_list[0], label_list[0]))[np.newaxis,:]
    
    return df_iou, offset_list_label_list, label_list
    
from keras.layers import Input
from keras import initializers

    
def build_RPN_model(feature_map_size):
    input_shape = (feature_map_size[0], feature_map_size[1], 512)
    k = 9
    input_ = Input(shape = input_shape)
    conv1 = Conv2D(512,
                   kernel_size = 3, 
                   padding = "same",
                   kernel_initializer = initializers.RandomNormal(stddev = 0.01), 
                   bias_initializer = initializers.Zeros())(input_)
    
    #delta regression
    regressor = Conv2D(4*k, 
                       kernel_size = 1, 
                       activation = 'linear',
                       name = 'delta_regression',
                       kernel_initializer=initializers.RandomNormal(stddev=0.01),
                       bias_initializer=initializers.Zeros())(conv1)
    
    #objectiveness score
    classifier = Conv2D(k*1,
                    kernel_size= 1,
                    activation= "sigmoid",
                    name="objectivess_score",
                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Zeros())(conv1)
    
    RPN = Model(inputs = [input_], outputs = [regressor, classifier])
    return RPN
    
    
"---------------------------------"
from tensorflow.keras import backend as K

""" CUSTOM LOSS FUNCTION """

def smooth_l1_loss(y_true, y_pred):
    """
    Calculates Smooth L1 loss
    """

    # Take absolute difference
    x = K.abs(y_true - y_pred)

    # Find indices of values less than 1
    mask = K.cast(K.less(x, 1.0), "float32")
    # Loss calculation for smooth l1
    loss = (mask * (0.5 * x ** 2)) + (1 - mask) * (x - 0.5)
    return loss


def custom_l1_loss(y_true, y_pred):
    """
    Regress anchor offsets(deltas) * only consider foreground boxes
    """
    offset_list= y_true[:,:,:-1]
    label_list = y_true[:,:,-1]
    
    # reshape output by the model
    y_pred = tf.reshape(y_pred, shape= (-1, 9*85*50, 4))
    
    positive_idxs = tf.where(K.equal(label_list, 1)) # select only foreground boxes
    
    # Select positive predicted bbox shifts
    bbox = tf.gather_nd(y_pred, positive_idxs)
    
    target_bbox = tf.gather_nd(offset_list, positive_idxs)
    loss = smooth_l1_loss(target_bbox, bbox)

    return K.mean(loss)

def custom_binary_loss(y_true, y_pred_objectiveness):
    '''
    Select both foreground and background class and compute cross entropy
    '''
    
    y_pred = tf.reshape(y_pred_objectiveness, shape= (-1, 9*85*50))
    y_true = tf.squeeze(y_true, -1)
    
    # Find indices of positive and negative anchors, not naeutral
    indices = tf.where(K.not_equal(y_true, -1)) # ignore -1 labels

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_match_logits = tf.gather_nd(y_pred, indices)
    anchor_class = tf.gather_nd(y_true, indices)
    
    
    # Cross entropy loss
    loss = K.binary_crossentropy(target=anchor_class,
                                output=rpn_match_logits
                                )
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    
    return loss
"------------------------------------"

""" CLIPPING, FILTERING AND NON-MAX SUPRESSION """
# now  that we have anchor deltas and objectiveness scores for each anchor box, we 
# adjust anchor boces, base on anchor deltas, further process them and filter boxes based on the following:
    #clip the proposals boxes to the image
    #remove the proposals who's height and width are less than some threshold value
    #sort all proposals based on the objectiveness score

def clip_coordinates(roi):
    roi[:, 0] = np.clip(roi[:, 0], 0, 1360)  # clip x1 coordinates
    roi[:, 1] = np.clip(roi[:, 1], 0, 800)  # clip y1 coordinates
    roi[:, 2] = np.clip(roi[:, 2], 0, 1360)  # clip x2 coordinates
    roi[:, 3] = np.clip(roi[:, 3], 0, 800)  # clip y2 coordinates
    return roi


def create_label_dict(label_file_path):
    label_dict = {}
    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            label_id, label_name = line.split(' ', 1)
            label_dict[int(label_id)] = label_name
    return label_dict


def apply_nms(roi, objectiveness_score, pre_NMS_topN=12000, n_train_post_nms=2000, min_size=8, iou_threshold=0.7):
    # Remove predicted boxes with either height or width < threshold.
    width = roi[:, 2] - roi[:, 0]  # xmax - xmin
    height = roi[:, 3] - roi[:, 1]  # ymax - ymin
    keep = np.where((width > min_size) & (height > min_size))[0]
    roi = roi[keep]
    score = objectiveness_score[:, keep]

    # Sort all (proposal, score) pairs by score from highest to lowest.
    sorted_idx = score.flatten().argsort()[::-1]
    score_sorted = score[:, sorted_idx]
    roi_sorted = roi[sorted_idx]

    # Select top N proposals (top 12000)
    score_sorted = score_sorted[:, :pre_NMS_topN]
    roi_sorted = roi_sorted[:pre_NMS_topN]
    
    # apply non-max supression on 12000 roi
    roi_idx = np.array(range(pre_NMS_topN))

    x1 = roi_sorted[:, 0]
    y1 = roi_sorted[:, 1]
    x2 = roi_sorted[:, 2]
    y2 = roi_sorted[:, 3]

    # Area of all roi
    # + 1 to prevent division by zero
    width_list = (x2 - x1) + 1
    height_list = (y2 - y1) + 1
    area_list = width_list * height_list

    # ROIs to keep as region proposals
    keep = []

    while roi_idx.size > 0:
        # Take the first ROI index
        current_id = roi_idx[0]

        # Add the current ROI to keep list
        keep.append(current_id)

        # Maximum of x1 of current and all other x1 ROI
        xx1 = np.maximum(x1[current_id], x1[roi_idx[1:]])

        # Maximum of y1 of current and all other y1 ROI
        yy1 = np.maximum(y1[current_id], y1[roi_idx[1:]])

        # Minimum of x2 of current and all other x2 ROI
        xx2 = np.minimum(x2[current_id], x2[roi_idx[1:]])

        # Minimum of y2 of current and all other y2 ROI
        yy2 = np.minimum(y2[current_id], y2[roi_idx[1:]])

        # Width of all the intersection area
        w = np.maximum(0., xx2 - xx1 + 1)

        # Height of all the intersection boxes
        h = np.maximum(0., yy2 - yy1 + 1)
        
        # Area of all the intersection boxes
        inter = w * h

        # IOU of current ROI and rest of the ROIs
        iou = inter / (area_list[current_id] + area_list[roi_idx[1:]] - inter)
        
        # Select boxes whose overlap is less than the threshold
        keep_idx = np.where(iou <= iou_threshold)[0]

        # Update the ROI index list (* note +1 to the indices list)
        roi_idx = roi_idx[keep_idx + 1]

    # Select only top 2000 proposals
    keep = keep[:n_train_post_nms]
    roi_sorted = roi_sorted[keep]
    score_sorted = score_sorted[:, keep]

    return roi_sorted, score_sorted

def label_rois(roi_sorted, gt_boxes_scaled, labels, label_dict_path, min_iou_threshold=0.1):
    gt_classes = create_label_dict(label_dict_path)

    # Find the IoU of all region proposals with all the ground-truth boxes
    iou_sorted = np.zeros((len(roi_sorted), len(gt_boxes_scaled)))

    # For each proposal
    for i, bbox in enumerate(roi_sorted):
        # For each ground truth box
        for j, gt_box in enumerate(gt_boxes_scaled):
            ymin, xmin, ymax, xmax = gt_box
            iou_sorted[i][j] = IOU(bbox, (xmin, ymin, xmax, ymax))

    # Get max IoU for each proposal
    max_iou_sorted = np.max(iou_sorted, axis=1)

    # Find the index of the ground truth box with the maximum IoU for each proposal
    gt_assign = np.argmax(iou_sorted, axis=1)

    # Filter out the proposals with IoU below the threshold
    gt_assign[max_iou_sorted < min_iou_threshold] = -1

    # Define the mapping from class name to integer labels
    class_to_label = {class_name: i for i, class_name in enumerate(gt_classes.values())}

    # Assign class labels to each proposal
    gt_roi_label = np.zeros(len(gt_assign), dtype=int)

    for i, label in enumerate(gt_assign):
        if label == -1:
            gt_roi_label[i] = -1
        else:
            class_name = gt_classes[label]
            gt_roi_label[i] = class_to_label[class_name]

    return gt_roi_label, iou_sorted



""" 
After generating RPN feature maps, use ROI Pooling to extract fixed-length features
from each proposal region
"""

from tensorflow.keras.layers import Layer


def apply_deltas_to_anchors(anchors, deltas):
    """
    Applies the deltas to the anchors to get the refined bounding boxes.

    Args:
        anchors: A numpy array of shape (num_anchors, 4) containing the coordinates of the anchors
        deltas: A numpy array of shape (num_anchors, 4) containing the predicted deltas

    Returns:
        A numpy array of shape (num_anchors, 4) containing the coordinates of the refined bounding boxes
    """
    # Convert anchors from (x1, y1, x2, y2) to (cx, cy, w, h)
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * widths
    cy = anchors[:, 1] + 0.5 * heights

    # Apply deltas to the (cx, cy, w, h) representation of the anchors
    cx += deltas[:, 0] * widths
    cy += deltas[:, 1] * heights
    widths *= np.exp(deltas[:, 2])
    heights *= np.exp(deltas[:, 3])

    # Convert (cx, cy, w, h) back to (x1, y1, x2, y2)
    x1 = cx - 0.5 * widths
    y1 = cy - 0.5 * heights
    x2 = cx + 0.5 * widths
    y2 = cy + 0.5 * heights

    # Stack the coordinates to form the refined bounding boxes
    refined_boxes = np.stack([x1, y1, x2, y2], axis=1)

    return refined_boxes

def clip_boxes_to_image(boxes, image_shape):
    """
    Clip boxes to image boundaries.

    Args:
        boxes: A numpy array of shape (N, 4) representing the coordinates of the boxes.
        image_shape: A tuple of integers representing the shape of the image.

    Returns:
        A numpy array of shape (N, 4) representing the clipped coordinates of the boxes.
    """
    height, width = image_shape[:2]
    # Clip the boxes to the image boundaries
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], width - 1)
    boxes[:, 3] = np.minimum(boxes[:, 3], height - 1)

    return boxes


################### ROIPooling & ProposalLayer ###################

class ProposalLayer(Layer):
    def __init__(self, proposal_count, nms_threshold, **kwargs):
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        super(ProposalLayer,self).__init__(**kwargs)
        
        def build(self,input_shape):
            super(ProposalLayer, self).build(input_shape)
        
        def call(self,inputs):
            # Extract inputs
            rpn_class = inputs[0]  # Objectness score
            rpn_bbox = inputs[1]  # Bounding box deltas
            anchors = inputs[2]  # Anchors generated by the RPN (xmin,ymin,xmax,ymax)

            # Determine number of anchors and proposals
            n_anchors = tf.shape(anchors)[1]
            n_proposals = self.proposal_count

            # Reshape the class and bbox predictions
            rpn_class = tf.reshape(rpn_class, [-1, 1])
            rpn_bbox = tf.reshape(rpn_bbox, [-1, 4])
            
            # Calculate deltas and proposal coordinates
            deltas = rpn_class * rpn_bbox
            proposals = apply_deltas_to_anchors(anchors, deltas)

            # Filter out-of-bound proposals
            proposals = clip_boxes_to_image(proposals, tf.shape(inputs[3]))

            # Non-max suppression
            indices = tf.image.non_max_suppression(proposals, rpn_class[:, 0],
                                               max_output_size=n_proposals,
                                               iou_threshold=self.nms_threshold)

            # Gather the top N proposals and return
            proposals = tf.gather(proposals, indices)
            padding = tf.maximum(n_proposals - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        
        def compute_output_shape(self,input_shape):
            return (None, self.proposal_count, 4)



class ROIPoolingLayer(Layer):
    def __init__(self, pooled_height, pooled_width, **kwargs):
        super(ROIPoolingLayer,self).__init__(**kwargs)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
    
    def compute_output_shape(self, input_shape):
        num_rois = input_shape[0]
        num_channels = input_shape[1]
        return (num_rois, self.pooled_height,self.pooled_width, num_channels)
    
    def call(self,inputs):
        feature_map = inputs[0]
        rois = inputs[1]
        
        #compute the ROI coordinates in feature map coordinates
        rois = tf.cast(rois,tf.float32)
        x1 = rois[:,0]
        y1 = rois[:,1]
        x2 = rois[:,2]
        y2 = rois[:,3]
        
        h = y2 - y1 #height of each roi box
        w = x2 - x1 #width of each roi box
        roi_level = tf.math.log(tf.sqrt(h * w) / 224.0) / tf.math.log(2.0)
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        
        #compute the pooled size of the roi
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        
        #divide the roi into a pooled grid
        pooled_h = []
        pooled_w = []
        
        for i in range(pooled_height):
            for j in range(pooled_width):
                pooled_h.append(tf.linspace(y1, y2, pooled_height + 1)[i])
                pooled_w.append(tf.linspace(x1, x2, pooled_width + 1)[j])
        
        pooled_h = tf.stack(pooled_h,axis=1)
        pooled_w - tf.stack(pooled_w,axis=1)
        
        #round pooled coordinates to the nearest integer
        pooled_h = tf.cast(tf.round(pooled_h), tf.int32)
        pooled_w = tf.cast(tf.round(pooled_w), tf.int32)
        
        #clip to feature map boundaries
        pooled_h = tf.maximum(tf.minimum(pooled_h, tf.shape(feature_map)[1] - 1), 0)
        pooled_w = tf.maximum(tf.minimum(pooled_w, tf.shape(feature_map)[2] - 1), 0)
        
        #gather pooled features from feature map
        pooled_features = tf.gather_nd(feature_map, tf.stack([tf.repeat(tf.range(tf.shape(rois)[0]), pooled_height * pooled_width), 
                                                             tf.tile(tf.reshape(pooled_h, [-1]), [tf.shape(rois)[0]]),
                                                             tf.tile(tf.reshape(pooled_w, [-1]), [tf.shape(rois)[0]])], axis=1))
        # Reshape pooled features into a fixed size tensor
        pooled_features = tf.reshape(pooled_features, [-1, pooled_height, pooled_width, tf.shape(feature_map)[-1]])
        return pooled_features





################### TEST ###################

dataset = tf.data.TFRecordDataset(example_proto)
dataset = dataset.map(parse_tfrecord).skip(1).take(1)

def main():     
    for image,bboxes,labels in dataset:
        image = tf.expand_dims(image,axis=0)
        #compute_centers(image, sizes, aspect_ratios, scales)
        #feature_map, feature_map_size = get_conv_feature_maps(image)
        
        """
        Get feature map + feature map size
        """
        feature_map, feature_map_size = get_feature_map(image)
        #boxes = generate_anchor_boxes(image,feature_map_size, image.shape[1:3])
        center_list = generate_center_points(image, feature_map_size,image.shape[1:3])
        
        #get ground truth bouding boxes
        gt_boxes = bboxes
        # Scale the bounding boxes to match the original image size
        gt_boxes_scaled = gt_boxes * [800, 1360, 800, 1360]
        gt_boxes_scaled = np.array(gt_boxes_scaled)
        
        anchor_list = generate_anchor_boxes2(gt_boxes,center_list, image, feature_map_size, image.shape[1:3])
        visualize_anchor_at_center_location(center_list,image, anchor_list, gt_boxes_scaled)
        
        
        """ VISUALZIE """
        
        #compute_iou_matrix(anchor_list, gt_boxes)
        df_iou,iou_matrix,valid_anchor_list = compute_iou_matrix3(anchor_list, gt_boxes_scaled)
        
        """"
        # Compute the IoU between every anchor box and every ground truth box
        iou = compute_iou_matrix2(boxes, gt_boxes_scaled)
        
        # Plot the IoU matrix
        plot_iou_matrix(iou)
        """
        img_data = image[0,...]
    
        top_anchors = sample_anchors(df_iou, iou_matrix, anchor_list)
        visualize_top_anchors(img_data, top_anchors, gt_boxes_scaled, anchor_list)
        df_iou = label_anchors(df_iou, top_anchors)
         
    
        df_iou = balance_anchors_RPN(df_iou)
         
        df_iou, offset_list_label_list, label_list = convert_corners_to_center(df_iou, anchor_list, valid_anchor_list, gt_boxes_scaled)
          
          
        RPN = build_RPN_model(feature_map_size)
         
        RPN.compile(loss = [custom_l1_loss, custom_binary_loss], optimizer= "adam")
          
        RPN.fit(feature_map,[offset_list_label_list, label_list], epochs= 100)
              
          
        #get the offsets and the objectiveness score
        anchor_deltas, objectiveness_score = RPN.predict(feature_map)
        print("Anchor Deltas shape is: " , anchor_deltas.shape)
        print("Objectiveness score shape is: ", objectiveness_score.shape)
         
         #shape both the predictions  
        n_anchors = len(ASPECT_RATIOS) * len(SCALES) * feature_map_size[0] * feature_map_size[1]
        anchor_deltas = anchor_deltas.reshape(-1, n_anchors, 4)
        print("Anchor Deltas reshape is: " , anchor_deltas.shape)
         
        objectiveness_score = objectiveness_score.reshape(-1,n_anchors)
        print("Objectiveness score reshape is: ", objectiveness_score.shape)
          
        # parse anchor deltas
        dx = anchor_deltas[:,:,0]
        dy = anchor_deltas[:,:,1]
        dw = anchor_deltas[:,:,2]
        dh = anchor_deltas[:,:,3]
         
        print(anchor_deltas.shape, objectiveness_score.shape)
         
        anchor_list = anchor_list.squeeze()
        anchor_list = np.expand_dims(anchor_list,0)
        print("Anchor_list_shape is: ",anchor_list.shape)
        
        #for each anchor box, convert coordinate format (min_x, min_y, max_x, max_y to height, width, center_x)
        anchor_width, anchor_height, anchor_center_x, anchor_center_y = to_center (anchor_list[0][:, 0], #x1
                                                                                   anchor_list[0][:, 1], #y1
                                                                                   anchor_list[0][:, 2], #x2
                                                                                   anchor_list[0][:, 3]) #y2
         
        #get the ROI
        roi_width, roi_height, roi_center_x, roi_center_y = adjust_deltas(anchor_width, 
                                                                           anchor_height, 
                                                                           anchor_center_x,
                                                                           anchor_center_y, 
                                                                           dx,
                                                                           dy,
                                                                           dw,
                                                                           dh)
        
        print(anchor_width.shape, roi_width.shape)
        
        #ROI format conversion
        roi_min_x, roi_min_y, roi_max_x, roi_max_y = to_VOC(roi_width, roi_height, roi_center_x, roi_center_y)
        roi = np.stack((roi_min_x, roi_min_y, roi_max_x, roi_max_y)).T
        roi = np.squeeze(roi)
        print(roi_min_x.shape, roi.shape)
        
        #clip predicted boxes to the image
        roi = clip_coordinates(roi)
        
        """" NMS """
        roi_sorted, score_sorted = apply_nms(roi, objectiveness_score)
        
        # visualizing top 20 anchor boxes
        img_ = np.copy(img_data)
        for i in range(0, 100):  
            x_min = int(roi_sorted[i][0])
            y_min = int(roi_sorted[i][1])
            x_max = int(roi_sorted[i][2])
            y_max = int(roi_sorted[i][3])
            print("First ROI: ", x_min, y_min, x_max, y_max)
            cv2.rectangle(img_,(x_min,y_min), (x_max,y_max), color = (255,0,0), thickness = 2)
    
        for i, bbox in enumerate(gt_boxes_scaled):
            ymin, xmin, ymax, xmax = bbox.astype(int)
            cv2.rectangle(img_, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)
    
        plt.imshow(img_)
        plt.show()
        
        labels = np.array(labels)
        print(labels)
        
        """ LABEL ROIS"""
        label_dict_path = "/Users/selmamusledin/Desktop/CV - Traffic Sign Detection/gtsdb.label.txt"
        
        gt_roi_label, iou_sorted = label_rois(roi_sorted, gt_boxes_scaled, labels, label_dict_path)
        
        max_iou_sorted = np.max(iou_sorted, axis=1)
    
        """ SAMPLE REGION PROPOSALS TO FEED TO ROI POOLING"""
        
        #threshold for foreground and background labels
        pos_threshold = 0.5
        neg_threshold_hi = 0.5
        neg_threshold_lo = 0.1
        
        #select proposals where iou is at least 50%
        keep_pos_idx_list = np.where(max_iou_sorted >= pos_threshold)[0]
        
        #select proposals whose IoU is less than 50% but greater or equal to lower limit 10%
        keep_neg_idx_list = np.where((max_iou_sorted < neg_threshold_hi) & (max_iou_sorted >=neg_threshold_lo))[0]
        
        # select 128 proposals of both classes (foreground and background)
        n_samples = 128
        neg_ratio = 0.75
        # number of foreground and background class
        n_foreground = int((1-neg_ratio) * n_samples)
        n_backgroud = int(neg_ratio * n_samples)
    
        # index of foreground and background ROIs
        foreground_index_list = keep_pos_idx_list
        background_index_list = keep_neg_idx_list
    
        # check if we have excessive foreground samples
        if len(foreground_index_list) > n_foreground:
            # randomly sample 32 foreground proposals
            select_pos_index_list = np.random.choice(foreground_index_list, n_foreground)
        else:
            select_pos_index_list = foreground_index_list
    
        # sample background examples if we don't have enough positive examples to match the anchor batch size
        if len(foreground_index_list) < n_foreground:
            diff = n_foreground - len(foreground_index_list)
            # add remaining value to background examples
            n_backgroud += diff
    
        # check if we have excessive background samples
        if len(background_index_list) > n_backgroud:
            # randomly sample remaining proposals as negative
            select_neg_index_list = np.random.choice(background_index_list, n_backgroud)
        else:
            select_neg_index_list = background_index_list
    
        # combine both the index list, foreground and background
        keep = np.hstack([select_pos_index_list, select_neg_index_list])
        
        # select corresponding proposals and labels
        sample_roi = roi_sorted[keep]
        gt_roi_labels = gt_roi_label[keep]
        # mark the background classes
        gt_roi_labels[select_pos_index_list.shape[0]:] = 0
        
        # """ visualizing foreground proposals """
        img_ = np.copy(img_data)
    
        for i in range(18):  
            # visualizing foreground proposals
            x_min = int(sample_roi[i][0])
            y_min = int(sample_roi[i][1])
            x_max = int(sample_roi[i][2])
            y_max = int(sample_roi[i][3])
            cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 
    
        for i, bbox in enumerate(gt_boxes_scaled):
            ymin,xmin,ymax,xmax = bbox
            cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(0,255, 0), thickness=3) 
    
        plt.imshow(img_)
        plt.show()
        
        """ Pass the selected ROIs to the ROIPooling layer"""
        pooling_layer = ROIPoolingLayer((7,7))
        
        #pass the feature map and the ROIs to the ROIPooling layer
        pooled_features = pooling_layer([feature_map, sample_roi])
        
        #flatten the pooled features to be passed to a fully connected layer
        flattened_features = tf.keras.layers.Flatten()(pooled_features)
        
        #add a FC layer to the network and pass the flattened features to it
        fc_layer = tf.keras.layers.Dense(4096, activation='relu')(flattened_features)
        
        #add another FC layer to the network
        fc_layer = tf.keras.layers.Dense(4096, activation = 'relu')(fc_layer)
        
        #add the output layer with a number of units equal to the number of classes
        NUM_CLASSES = 42
        output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(fc_layer)
        
        
        
    
    ################### TEST ###################

main()


        
        
        
   
    
    
   


  