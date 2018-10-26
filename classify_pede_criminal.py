import numpy as np
import time
import math
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
#from keras.utils import to_categorical
import pickle
import cPickle

import random

#This Keras doesn't support this.
def to_categorical(label, num_classes):
    n = label.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), label] = 1.0
    return one_hot

def getCorrespoindingFeatureAndLabel(feat, dets, Label, IsLabeled, thre = 0.5): # this thre has to be the same with in function vis_detections in label_tool.py.
    # Input: feat, dets, Label are all the detections of faster rcnn, a lot with score < thre. In Label, we have labeled them via label_tool.py, and all those < thre are kept as -1.
    # Output: feature : fc7 layer's feature of all the bboxes with score > thre. 
    #         label   : the corresponding label, or equivalently, label = Label[not -1].

    num_images = len(feat)
    feature = []
    label = []
    for i in xrange(num_images):
        inds = np.where(dets[i][:, -1] >= thre)[0]
        if len(inds) == 0 or not IsLabeled[i]:
            continue
        feat_this_img = feat[i][inds,:]
        feature.append(feat_this_img)
        label_this_img = Label[i][inds]
        label.append(label_this_img)
    return feature, label

#Resize the image
def DataPreprocess(feature, label):
    # Arrange in features and labels format.
    X = []
    Y = []
    num_images = len(feature)
    for i in xrange(num_images):
        for box in xrange(feature[i].shape[0]):
            X.append(feature[i][box,:])
            Y.append(label[i][box])
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Normalize to [0,1]
    Xmax = np.max(X)
    Xmin = np.min(X)
    X = (X - Xmin)/(Xmax - Xmin)

    # Balance 3 classes. 0: not a human 1: pedestrain 2: criminal
    cnt0 = np.count_nonzero(Y==0)
    cnt1 = np.count_nonzero(Y==1)
    cnt2 = np.count_nonzero(Y==2)
    ind0 = np.where(Y==0)[0]
    ind1 = np.where(Y==1)[0]
    ind2 = np.where(Y==2)[0]
    #print cnt0, cnt1, cnt2, X.shape
    min_cnt = min(min(cnt0,cnt1),cnt2)
    X0 = X[ind0[:min_cnt],:]; X1 = X[ind1[:min_cnt],:]; X2 = X[ind2[:min_cnt],:]
    Y0 = Y[ind0[:min_cnt]]; Y1 = Y[ind1[:min_cnt]]; Y2 = Y[ind2[:min_cnt]]


    # Split train/test
    split = 0.8 
    x_train = np.vstack((X0[:int(split*min_cnt),:], X1[:int(split*min_cnt),:], X2[:int(split*min_cnt),:]))
    y_train = np.hstack((Y0[:int(split*min_cnt)], Y1[:int(split*min_cnt)], Y2[:int(split*min_cnt)]))
    x_test = np.vstack((X0[int(split*min_cnt):,:], X1[int(split*min_cnt):,:], X2[int(split*min_cnt):,:]))
    y_test = np.hstack((Y0[int(split*min_cnt):], Y1[int(split*min_cnt):], Y2[int(split*min_cnt):]))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    return x_train, y_train, x_test, y_test


def Classify(x_train, y_train, x_test, y_test):
    # convert label to one-hot
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    #print y_train, y_test, y_train.shape, y_test.shape

    #print x_train, x_test, x_train.shape, x_test.shape

    model = Sequential()
    # Layer1: FC
    model.add(Dense(output_dim = 100, input_dim = 4096, activation = 'relu'))
    model.add(Dropout(0.5))
    # Layer2: FC + Softmax
    model.add(Dense(output_dim = 3, activation = 'softmax'))
    sgd = SGD(lr = 0.005, decay = 1e-6, momentum=0.9,nesterov=True)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
    checkpointer = ModelCheckpoint(filepath = 'weights.hdf5', verbose=1, save_best_only=True)

    # Training the model
    model.fit(x_train, y_train, batch_size = 20, nb_epoch = 300, validation_split = 0.2, verbose = 1, callbacks=[checkpointer])

    testPredict = model.predict(x_test)

    yPredict = np.zeros((testPredict.shape[0],))

    print testPredict
    for i in xrange(testPredict.shape[0]):
        yPredict[i] = np.argmax(testPredict[i,:])

    print yPredict
    with open('testPredict.pkl', 'wb') as f:
        pickle.dump(testPredict, f)
    return yPredict

#The main function
if __name__ == '__main__':
    # Reproducity
    np.random.seed(3)
    # load the features
    # feat[class_index][image_index] is 2d array: num_bbox * 4096
    feat_file = '../fc7_features.pkl'
    with open(feat_file) as f:
        feat = pickle.load(f)
    feat = feat[2] # We only need the 'others' feature.

    # load detection file
    with open('../detections.pkl', 'rb') as f:
        dets = pickle.load(f)
    dets = dets[2]

    # Load latest label process, IsLabeled[i] == 1 means the i th image is labeled.
    with open('../IsLabeled.pkl','rb') as f:
        IsLabeled = cPickle.load(f)
    with open('../Pede_Criminal_Label.pkl', 'rb') as f:
        Label = cPickle.load(f)
    # Label[img_index] is num_bbox * 1 array, 0: not even a human 1: pedestrain 2: criminal

    feature, label = getCorrespoindingFeatureAndLabel(feat, dets, Label, IsLabeled)
    #print len(feature), len(label)
    x_train, y_train, x_test, y_test = DataPreprocess(feature, label)
    print x_train.shape, x_test.shape, y_train.shape, y_test.shape, np.count_nonzero(y_train==0), np.count_nonzero(y_test==0)
    yPredict = Classify(x_train, y_train, x_test, y_test)
    precision = np.count_nonzero(yPredict==y_test)/float(y_test.shape[0])
    print 'Precision:', precision

    
    
