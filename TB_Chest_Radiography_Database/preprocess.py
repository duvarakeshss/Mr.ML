#Importing the necessary libraries:
import cv2 as cv
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

#Initializing the values needed for all the image files
normaldir = './Normal'
tbdir = './Tuberculosis'
images = []
labels = []
imagesize = 256

#Storing all the image directories in the 'images' array and corresponding them to either 1 for TB images or 0 for normal images.
for x in os.listdir(normaldir):
    imagedir = os.path.join(normaldir, x)
    image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (imagesize, imagesize))
    images.append(image)
    labels.append(0)
    
for y in os.listdir(tbdir):
    imagedir = os.path.join(tbdir, y)
    image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (imagesize, imagesize))
    images.append(image)
    labels.append(1)

#Converting to NumPy arrays since they have more features than regular lists
images = np.array(images)
labels = np.array(labels)

#Splitting the images and labels into training and testing sets, then normalizing the values within them for computational efficiency (from 0-255 scale to 0-1 scale)
imagetrain, imagetest, labeltrain, labeltest = train_test_split(images, labels, test_size=0.3, random_state=42)
imagetrain = (imagetrain.astype('float32'))/255
imagetest = (imagetest.astype('float32'))/255

#Flattening the image array into 2D (making it [2940 images] x [all the pixels of the image in just one 1D array]) to be suitable for SMOTE oversampling
imagetrain = imagetrain.reshape(2940, (imagesize*imagesize))

#Performing oversampling
smote = SMOTE(random_state=42)
imagetrain, labeltrain = smote.fit_resample(imagetrain, labeltrain)

#Unflattening the images now to use them for convolutional neural network (4914 images of 256x256 size, with 1 color channel (grayscale, as compared to RGB with 3 color channels))
imagetrain = imagetrain.reshape(-1, imagesize, imagesize, 1)
print(imagetrain.shape)

#Classes balanced - equal counts of each label
print(np.unique(labeltrain, return_counts=True))


