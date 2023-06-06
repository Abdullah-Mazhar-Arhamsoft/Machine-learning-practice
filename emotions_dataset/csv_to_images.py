# This script takes a CSV file containing facial expression data from the FER 2013 dataset and
# converts the grayscale images into 3-channel images so that they can be used as input for
# convolutional neural networks (CNNs) designed for RGB images. The script requires two command line
# parameters: the path to the CSV file and the output directory. It generates the images and saves
# them in three directories inside the output directory - Training, PublicTest, and PrivateTest, which
# are the three original splits in the dataset.
'''
This script creates 3-channel gray images from FER 2013 dataset.

It has been done so that the CNNs designed for RGB images can 
be used without modifying the input shape. 

This script requires two command line parameters:
1. The path to the CSV file
2. The output directory

It generates the images and saves them in three directories inside 
the output directory - Training, PublicTest, and PrivateTest. 
These are the three original splits in the dataset. 
'''

import os
import csv
import argparse
import numpy as np 
import scipy.misc
import imageio

# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--file', required=True, help="path of the csv file")
# parser.add_argument('-o', '--output', required=True, help="path of the output directory")
# args = parser.parse_args()

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1

with open("emotions_dataset/emotions.csv") as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    next(datareader,None)
	
    for row in datareader:
        
        emotion = row[0]
        pixels = row[1].split()
        usage = row[2]
        pixels_array = np.asarray(pixels, dtype=np.int_)

        image = pixels_array.reshape(w, h)
        #print image.shape

        stacked_image = np.dstack((image,) * 3).astype(np.uint8)
        #print stacked_image.shape

        image_folder = os.path.join("images", usage)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file = os.path.join(image_folder, f'{str(id)}_{emotion}.jpg')
        imageio.imwrite(image_file, stacked_image)
        id += 1 
        if id % 100 == 0:
            print(f'Processed {id} images')

print(f"Finished processing {id} images")
