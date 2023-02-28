import read_data
import tensorflow as tf
import pandas as pd
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np


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

    example = tf.io.parse_single_example(example_proto, feature_description)

    height = example['image/height']
    width = example['image/width']
    filename = example['image/filename']
    source_id = example['image/source_id']
    encoded_image = example['image/encoded']
    image_format = example['image/format']
    xmin = example['image/object/bbox/xmin']
    xmax = example['image/object/bbox/xmax']
    ymin = example['image/object/bbox/ymin']
    ymax = example['image/object/bbox/ymax']
    classes_text = example['image/object/class/text']
    classes = example['image/object/class/label']

    return height, width, filename, source_id, encoded_image, image_format, xmin, xmax, ymin, ymax, classes_text, classes

def plot_heatmap(height, width, encoded_image, xmin, ymin, xmax, ymax):
    #DECODE -> RESIZE -> NORMALIZE
    image = tf.image.decode_jpeg(encoded_image)
    image = tf.image.resize(image,[height,width])
    image = tf.cast(image, tf.float32) /255.0
    
    #create a mask to visualize the bounding boxes
    mask = np.zeros((height,width,3))
    mask[:] = (0,0,128)
    
    xmin = tf.sparse.to_dense(xmin)
    ymin = tf.sparse.to_dense(ymin)
    xmax = tf.sparse.to_dense(xmax)
    ymax = tf.sparse.to_dense(ymax)
    
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
    xmax = tf.cast(xmax, tf.float32)
    ymax = tf.cast(ymax, tf.float32)
    width = tf.cast(width, tf.float32)
    height = tf.cast(height,tf.float32)
    
    for i in range(xmin.shape[0]):
        x0 = int(xmin[i] * width)
        y0 = int(ymin[i] * height)
        x1 = int(xmax[i] * width)
        y1 = int(ymax[i] * height)
        mask[y0:y1, x0:x1] = (255,0,0)
        
    plt.imshow(image)
    plt.imshow(mask,cmap='hot')
    plt.show()

    
def main():
    
    # get label information 
    #label_file = read_data.label_to_dict(read_data.LABEL_PATH)
    #print(gt_label[3])
    
    #read_data.csv_record(read_data.ANNOTS_TRAIN, label_file, 'train.record')
    #read_data.csv_record(read_data.ANNOTS_VAL, label_file, 'val.record')
    
    # file_path = "/Users/selmamusledin/Desktop/CV - Traffic Sign Detection/TFRecords/train.record"
    # dataset = tf.data.TFRecordDataset(file_path)
    # dataset = dataset.map(parse_tfrecord)

    # for height, width, filename, source_id, encoded_image, image_format, xmin, xmax, ymin, ymax, classes_text, classes in dataset.take(2):
    #     print("height:", height)
    #     print("width:", width)
    #     print("filename:", filename)
    #     print("source_id:", source_id)
    #     #print("encoded_image:", encoded_image)
    #     print("image_format:", image_format)
    #     print("xmin:", xmin)
    #     print("xmax:", xmax)
    #     print("ymin:", ymin)
    #     print("ymax:", ymax)
    #     print("classes_text:", classes_text)
    #     plot_heatmap(height, width, encoded_image, xmin, ymin, xmax, ymax)

    # print('Showing heatmap for given image: ')
    # tfrecord_file = tf.data.TFRecordDataset("/Users/selmamusledin/Desktop/CV - Traffic Sign Detection/train/train.csv")
    # for example_proto in tfrecord_file.take(2):
    #     height, width, filename, source_id, encoded_image, image_format, xmin, xmax, ymin, ymax, classes_text, classes = parse_tfrecord(example_proto)
    #     plot_heatmap(height, width, encoded_image, xmin, ymin, xmax, ymax)
    print('Main')

if __name__ == '__main__': 
    main() 


