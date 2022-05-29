#!/usr/bin/env python
# coding: utf-8

##Downloading and unpacking the dataset
# get_ipython().system('wget https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz')
# get_ipython().system('tar -xf mnist_png.tar.gz')


# Importing the required libraries
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import pickle

##Combining the train and test data
# get_ipython().system('cp -r /content/mnist_png/testing/* /content/mnist_png/training/')
# get_ipython().system('rm -rf /content/mnist_png/testing /content/mnist_png.tar.gz')


# Preparing the dataset
folder_0 = '/home/sam/digit-interface/example/Data_2/_60/trial_0_gray'
folder_1 = '/home/sam/digit-interface/example/Data_2/_60/trial_1_gray'
image_path = []
for idx, dir in enumerate([folder_0, folder_1]):
    for file in os.listdir(dir):
        image_path.append((idx, os.path.join(dir, file)))


def main(thresh):
    t0 = time.time()

    def CalcFeatures(img, th):

        sift = cv2.SIFT_create(th)
        # kp = sift.detect(img, None)
        # img = cv2.drawKeypoints(gimg, kp, img)
        # plt.imshow(img)

        # sift = cv2.xfeatures2d.SIFT_create(th)
        kp, des = sift.detectAndCompute(img, None)
        return des

    '''
    All the files appended to the image_path list are passed through the
    CalcFeatures functions which returns the descriptors which are 
    appended to the features list and then stacked vertically in the form
    of a numpy array.
    '''

    features = []
    for idx, file in image_path:
        img = cv2.imread(file, 0)
        img_des = CalcFeatures(img, thresh)
        if img_des is not None:
            features.append(img_des)
    features = np.vstack(features)

    '''
    K-Means clustering is then performed on the feature array obtained 
    from the previous step. The centres obtained after clustering are 
    further used for bagging of features.
    '''

    k = 150
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centres = cv2.kmeans(features, k, None, criteria, 10, flags)

    '''
    The bag_of_features function assigns the features which are similar
    to a specific cluster centre thus forming a Bag of Words approach.  
    '''

    def bag_of_features(features, centres, k=500):
        vec = np.zeros((1, k))
        for i in range(features.shape[0]):
            feat = features[i]
            diff = np.tile(feat, (k, 1)) - centres
            dist = pow(((pow(diff, 2)).sum(axis=1)), 0.5)
            idx_dist = dist.argsort()
            idx = idx_dist[0]
            vec[0][idx] += 1
        return vec

    labels = []
    vec = []
    for idx, file in image_path:
        img = cv2.imread(file, 0)
        img_des = CalcFeatures(img, thresh)
        if img_des is not None:
            img_vec = bag_of_features(img_des, centres, k)
            vec.append(img_vec)
            labels.append(idx)
    vec = np.vstack(vec)

    '''
    Splitting the data formed into test and split data and training the 
    SVM Classifier.
    '''

    X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2)
    clf = SVC()
    clf.fit(X_train, y_train)

    """
    # dump model
    import pickle
    pickle.dump(clf, open('model.pkl', 'wb'))
    
    # load model and predict
    clf = pickle.load(open('model.pkl', 'rb'))
    clf.predict(input)
    """

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    conf_mat = confusion_matrix(y_test, preds)

    t1 = time.time()

    # output model
    with open('sift_svc.pkl', 'wb') as f:
        pickle.dump(clf, f)
        print('Saved model to sift_svc.pkl')

    return acc * 100, conf_mat, (t1 - t0)


accuracy = []
timer = []
# for i in range(5, 26, 5):
for i in {5, }:
    print('\nCalculating for a threshold of {}'.format(i))
    data = main(i)
    accuracy.append(data[0])
    conf_mat = data[1]
    timer.append(data[2])
    print('\nAccuracy = {}\nTime taken = {} sec\nConfusion matrix :\n{}'.format(data[0], data[2], data[1]))

""" result
Calculating for a threshold of 5

Accuracy = 99.87460815047022
Time taken = 506.7863178253174 sec
Confusion matrix :
[[825   0]
 [  2 768]]

Calculating for a threshold of 10

Accuracy = 99.74921630094043
Time taken = 527.0500175952911 sec
Confusion matrix :
[[774   0]
 [  4 817]]

Calculating for a threshold of 15

Accuracy = 99.68652037617555
Time taken = 522.825493812561 sec
Confusion matrix :
[[812   3]
 [  2 778]]

Calculating for a threshold of 20

Accuracy = 99.49843260188088
Time taken = 526.8498702049255 sec
Confusion matrix :
[[771   3]
 [  5 816]]

Calculating for a threshold of 25

Accuracy = 99.68652037617555
Time taken = 493.0617265701294 sec
Confusion matrix :
[[801   0]
 [  5 789]]

"""


class SIFTSVC(object):

    def __init__(self):
        # sift
        self.nfeatures = 5

        self.model = None

    def train(self):
        pass

    def test(self):
        pass

    def load_model(self, path):
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def save_model(self, model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def extract_feature(self, img_path):
        img = cv2.imread(img_path, 0)  # 0 means loading as gray image
        sift = cv2.SIFT_create(self.nfeatures)
        kp, des = sift.detectAndCompute(img, None)
        return des


    @staticmethod
    def process_feature(img_des, features):
        k = 150
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centres = cv2.kmeans(features, k, None, criteria, 10, flags)

        img_vec = SIFTSVC.bag_of_features(img_des, centres, k)

        return img_vec

    @staticmethod
    def bag_of_features(features, centres, k=500):
        vec = np.zeros((1, k))
        for i in range(features.shape[0]):
            feat = features[i]
            diff = np.tile(feat, (k, 1)) - centres
            dist = pow(((pow(diff, 2)).sum(axis=1)), 0.5)
            idx_dist = dist.argsort()
            idx = idx_dist[0]
            vec[0][idx] += 1
        return vec
