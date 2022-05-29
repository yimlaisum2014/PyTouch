from __future__ import print_function

import os

import keras
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, maximum
from keras.layers import Conv2D, MaxPooling2D, Reshape, Lambda
from keras import backend as K
from scipy import misc
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from sklearn import svm
import tools
import cv2

batch_size = 128
num_classes = 2
epochs = 100
patch_size = 32

# input image dimensions
img_rows, img_cols = 640, 480

# # the data, shuffled and split between train and test sets
# #read in training data
# class_num = 2
# train_ele_num = 52
# x_train = []
# y_train = []
# for i in range(class_num):
# 	for j in range(train_ele_num):
# 		filename = 'selective_32/data/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg'
# 		img = misc.imread(filename)
# 		img_flattened = np.reshape(img, img.shape[0]*img.shape[1])
# 		x_train.append(img_flattened)
# 		y_train.append(i)
# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
#
# #read in testing data
# ele_num = 2
# x_test = []
# y_test = []
# for i in range(class_num):
# 	for j in range(train_ele_num, ele_num):
# 		filename = 'selective_32/data/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg'
# 		img = misc.imread(filename)
# 		img_flattened = np.reshape(img, img.shape[0]*img.shape[1])
# 		x_test.append(img_flattened)
# 		y_test.append(i)
# x_test = np.asarray(x_test)
# y_test = np.asarray(y_test)

folder_0 = '/home/sam/digit-interface/example/Data_2/_60/trial_0_gray'
folder_1 = '/home/sam/digit-interface/example/Data_2/_60/trial_1_gray'

vec = []
labels = []
for idx, dir in enumerate([folder_0, folder_1]):
    for file in os.listdir(dir):
        filename = os.path.join(dir, file)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_flattened = np.reshape(img, img.shape[0] * img.shape[1])
        vec.append(img_flattened)
        labels.append(idx)

x_train, x_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2)

x_train, x_test = np.array(x_train), np.array(x_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, patch_size, patch_size)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (patch_size, patch_size, 1)
#

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

input = Input(shape=input_shape)

# hidden layer 0
left = Conv2D(filters=96, kernel_size=(8, 8), activation='relu', padding='same', name='h0_l')(input)
right = Conv2D(filters=96, kernel_size=(8, 8), activation='relu', padding='same', name='h0_r')(input)
h0 = maximum([left, right])
# apply max pooling
h0 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='pool0')(h0)

# hidden layer 1
left = Conv2D(filters=192, kernel_size=(8, 8), activation='relu', padding='same', name='h1_l')(h0)
right = Conv2D(filters=192, kernel_size=(8, 8), activation='relu', padding='same', name='h1_r')(h0)
h1 = maximum([left, right])
# apply max pooling
h1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='pool1')(h1)

# hidden layer 2
left = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same', name='h2_l')(h1)
right = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same', name='h2_r')(h1)
h2 = maximum([left, right])
# apply max pooling
h2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h2)

# maxout layer
# x = Lambda(lambda x: K.max(h2, 3, True))(h2)
x_shape = h2.get_shape().as_list()
x = Reshape((x_shape[1] * x_shape[2] * x_shape[3],))(h2)
print(x.shape)
m1 = Dense(500, name='dense1')(x)
m2 = Dense(500, name='dense2')(x)
m3 = Dense(500, name='dense3')(x)
m4 = Dense(500, name='dense4')(x)
m5 = Dense(500, name='dense5')(x)
maxout = maximum([m1, m2, m3, m4, m5])
final = Model(input, maxout)
print(final.summary())
# final.load_weights('macro_48_48_model.h5', by_name=True)
x_train_grids = [tools.random_image_crop(im, patch_size) for im in x_train]
train_features = final.predict(np.asarray(x_train_grids))

clf = svm.LinearSVC()
print(train_features.shape)
print(np.ravel(y_train).shape)
clf.fit(train_features, np.ravel(y_train))
print('training finished.')

x_test_grids = [tools.extract_grid_patches(im, patch_size) for im in x_test]
final_prediction = []
for grids in x_test_grids:
    test_features = final.predict(grids)
    predictions = clf.predict(test_features).tolist()
    final_prediction.append(max(set(predictions), key=predictions.count))
y_test = np.ravel(y_test)
print('accuracy:' + str(sum(y_test == final_prediction) / y_test.shape[0]))
