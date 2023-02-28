
"""
FASTER R-CNN
    1. Region Proposal Network
"""

"""STEP 1 - DATA PREPROCESSING"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle             #library used for serializing and deserializing a Python structure
import os
import cv2
import csv
from tensorflow.keras.preprocessing.image import img_to_array, load_img

####### SET PATHS #######
BASE_PATH = "/Users/selmamusledin/Desktop/CV - Traffic Sign Detection"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "GTSDB"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "gt.txt"])

TRAIN_IMAGES = os.path.sep.join([BASE_PATH, 'train'])
TEST_IMAGES = os.path.sep.join([BASE_PATH, 'test'])
TARGET_DIRECTORY = os.path.sep.join([BASE_PATH, 'input_img'])

ANNOTS_TRAIN = os.path.sep.join([TRAIN_IMAGES, 'train.csv'])
ANNOTS_VAL = os.path.sep.join([TRAIN_IMAGES, 'val.csv'])
ANNOTS_TEST = os.path.sep.join([TRAIN_IMAGES,'test.csv'])

LABEL_PATH = "/Users/selmamusledin/Desktop/CV - Traffic Sign Detection/gtsdb.label.txt"  #label of each corresponding id

save_img = True

####### CREATE DIRECTORIES #######
if not os.path.isdir(TRAIN_IMAGES):
    os.makedirs(TRAIN_IMAGES)
    
if not os.path.isdir(TEST_IMAGES):
    os.makedirs(TEST_IMAGES)
    
if not os.path.isdir(TARGET_DIRECTORY):
    os.makedirs(TARGET_DIRECTORY)


""" GTSDB DATASET ANALYSIS """                                                                     #get current working directory
data = pd.read_csv("gt.csv",sep=';',header = None)
data.columns = ['img','x1','y1','x2','y2','id']

print("Number of image files:", len(data['img'].unique()))
print("Number of traffic sign classes:", len(data['id'].unique()))
print("Number of traffic sign instances", data['id'].count())

#print(pd.value_counts(data['id'],sort=False))
data['id'].hist(bins=86)

""" STRATIFIED SHUFFLE SPLIT """
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def stratifiedShuffleSplit(data,test_size=0.3,thresh=1):
    y_less = data.groupby('id').filter(lambda x: len(x) <= thresh)  #I want to filter where there's only one occurence of a traffic sign 
    data = pd.concat([data,y_less], ignore_index = True)
    
    #StratifiedShuffleSplit provides train/test indices to split data in train/test
    iterator_obj = StratifiedShuffleSplit(n_splits = 1, test_size=test_size, random_state=42)  # sss returns an iterator object that is used to generate indices for training and testing 
    #train_index, test_index = list(*iterator_obj)                       # extract elements of the generator into two separate variables, train and test
                                                                        # train_index, test_index = next(iterator_obj)
    train_index, test_index = next(iterator_obj.split(data['id'],data['id']))
    
    # I have the index -> build train set from index location
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]
      
    return train_set, test_set


# x_train -> 0.7 of data, x_test(temp) is 0.3 of data
x_train, temp = stratifiedShuffleSplit(data,0.3,1)    

# I split the x_test(temp) into testing and validation in half
x_test, x_val = stratifiedShuffleSplit(temp,0.5,1)

plt.hist([x_train['id'],x_test['id'],x_val['id']], stacked=True,label=["Train","Test","Val"], bins=86)
plt.show()

#set image resolution for each training image
x_train["width"] = 1360
x_train["height"] = 800    

#set image resolution for each testing image                               
x_test["width"] = 1360
x_test["height"] = 800    

#set image resolution for each validation image
x_val["width"] = 1360
x_val["height"] = 800       


x_train.to_csv(os.path.sep.join([TRAIN_IMAGES,"train.csv"]), index = False)
x_test.to_csv(os.path.sep.join([TEST_IMAGES,"test.csv"]),index=False)
x_val.to_csv(os.path.sep.join([TRAIN_IMAGES,"val.csv"]),index=False)




""" SPLIT IMAGES """
import hashlib
import random
import shutil
import tensorflow as tf
from tqdm import tqdm
from PIL import Image, ImageDraw,ImageFont
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple,OrderedDict
import configparser

def TFRecordCreator(input_list, out):
    writer = tf.io.TFRecordWriter(out)
    for tf_example in input_list:
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Record File " + str(out) + " has been successfully created!")


def SaveImage(image, col, filename, label_file):   #saves image with all objectd detected
    #this gives me an index and the row
    for index,row in col.object.iterrows(): #i'm taking the whole object specified by the id
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        label = label_file[int(row['id'])]
        
        bbox = (x1,y1), (x2,y2)
        drawing_object = ImageDraw.Draw(image)
        drawing_object.rectangle(bbox,outline='red')
        drawing_object.text([x1+20,y1+20], label, color='red')
    
    save_dir = TARGET_DIRECTORY
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    image.save(os.path.sep.join([save_dir,filename]))


def label_to_dict(label_path):
    label_dict = {}
    with open(label_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if not line.split(): #splits string into list at whitespace
                continue
            line = line.strip()
            label,filename = line.split(' ',1) #1 split
            print(label,filename)
            label_dict[int(label)] = filename
    return label_dict

import io
# an Example is a standard proto storing data for training and inference
def dict_TFExample(img_path,col,label_file,ignore_difficult_instances=False):
    
    with tf.io.gfile.GFile(img_path,'rb') as f:
        encoded_jpg = f.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size
    
    filename = (col.img[:-3] + 'png')
    image_format = b'png'
    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]
    classes=[]
    classes_text=[]
    
    for index, row in col.object.iterrows():
        xmin.append(row['x1']/width)
        xmax.append(row['x2']/width)
        ymin.append(row['y1']/height)
        ymax.append(row['y2']/height)
        classes.append(int(row['id']+1))
        classes_text.append(label_file[int(row['id'])].encode('utf8'))
    
    if save_img:    
        SaveImage(image, col, filename, label_file)

    #load all objects in a picture into example
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example



def split(df, col):
    data = namedtuple('data', ['img', 'object'])
    gb = df.groupby(col)
    return [data(img, gb.get_group(x)) for img, x in zip(gb.groups.keys(), gb.groups)] #filename with picture



def csv_record(csv, label_path, out='out.record'):
    examples = pd.read_csv(csv)
    grouped = split(examples, 'img') #img is the filename
    out_examples = []
    
    
    #print(grouped[:5])
    for group in grouped:
        img_path = os.path.sep.join([os.path.sep.join([BASE_PATH,'png_images']), group.img]) #get filename of each image
        img_path = img_path[:-3] + "png" #get png filename of each image
        #print(group.img)
        tf_example = dict_TFExample(img_path, group, label_path)
        out_examples.append(tf_example)
    
    #create a directory where I will store the records of examples
    output_dir = os.path.sep.join([BASE_PATH, 'TFRecords'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = os.path.sep.join([output_dir, out])
    TFRecordCreator(out_examples, output)
    print(out + ' has been successful created')



""" CREATE PBTXT """
PB_TEXT = os.path.sep.join([BASE_PATH,"gtsdb.pbtxt"])
with open(PB_TEXT,"w+") as output:
    with open(os.path.sep.join([BASE_PATH,"gtsdb.label.txt"])) as f:
        lines = f.readlines()
        for line in lines:
            if not line.split():
                continue
            line = line.strip()
            number, name = line.split(' ',1)
            output_str = "item { \n" + \
                        "  id: " + number + "\n" + \
                        "  name: \'" + name + '\'' + \
                        "\n}\n\n"

            output.write(output_str)
print('Creating pbtxt was a success')

# """ PPM TO PNG """
# PNG_IMAGE_PATH = os.path.sep.join([BASE_PATH,"png_images"])

# if os.path.exists(PNG_IMAGE_PATH):
#     shutil.rmtree(PNG_IMAGE_PATH)
# os.makedirs(PNG_IMAGE_PATH)

# for filename in os.listdir(TRAIN_IMAGES):
#     if filename[-3:] == "ppm":
#         img_path = os.path.sep.join([TRAIN_IMAGES,filename])
#         img = Image.open(img_path)
#         png_path = os.path.sep.join([PNG_IMAGE_PATH, filename[:-3]+"png"])
#         print(png_path)
#         img.save(png_path)


# =============================================================================
# # move images from data into corresponding directories
# def move_images(output_path, annotFilename):
#     csvFile = pd.read_csv(os.path.sep.join([output_path,annotFilename]))
#     for index,row in csvFile.iterrows():
#         filename = row['img']
#         print(filename)
#         
#         #take the path of the corresponding image in each annot.csv
#         img_path = os.path.sep.join([IMAGES_PATH, filename])
#         
#         #check if the image exists in the source directory
#         if os.path.exists(img_path):    
#             img_out_path = os.path.sep.join([output_path, filename])
#             shutil.copyfile(img_path,img_out_path)
#         
#     
# move_images(TRAIN_IMAGES,"train.csv")
# move_images(TEST_IMAGES,"test.csv")
# move_images(TRAIN_IMAGES,"val.csv")    
#     
# =============================================================================

# for filename in os.listdir(TEST_IMAGES):
#     if filename[-3:] == "ppm":
#         img_path = os.path.sep.join([TEST_IMAGES,filename])
#         img = Image.open(img_path)
#         png_path = os.path.sep.join([TEST_IMAGES, filename[:-3]+"png"])
#         print(png_path)
#         img.save(png_path)
#         os.remove(img_path)
    



    
