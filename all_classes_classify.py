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

#This Keras doesn't support this.
def to_categorical(label, num_classes):
    n = label.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), label] = 1.0
    return one_hot

#Resize the image
def DataPreprocess(features, num_classes = 3):
    # Arrange in features and labels format.
    X = []
    Y = []
    num_images = len(feat[0])
    for j in xrange(1,num_classes):
        for i in xrange(num_images):
            for box in xrange(feat[j][i].shape[0]):
                X.append(feat[j][i][box,:])
                Y.append(j-1)
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Balance 2 classes. In our case police is about 0.5*others
    cnt = np.count_nonzero(Y==0)
    X = X[:2*cnt]
    Y = Y[:2*cnt]

    # Normalize to [0,1]
    Xmax = np.max(X)
    Xmin = np.min(X)
    X = (X - Xmin)/(Xmax - Xmin)

    # Split train/test
    split = 0.9
    train_index = list(range(int(round(split*cnt)))) + list(range(cnt,int(round((1+split)*cnt))))
    test_index = list(range(int(round(split*cnt)), cnt)) + list(range(int(round((1+split)*cnt)),2*cnt))
    x_train = X[train_index,:]
    x_test = X[test_index,:]
    y_train = Y[train_index]
    y_test = Y[test_index]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #y_train = y_train.astype('float32')
    #y_test = x_test.astype('float32')
    return x_train, y_train, x_test, y_test


def Classify(x_train, y_train, x_test, y_test):
    # convert label to one-hot
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    #print y_train, y_test, y_train.shape, y_test.shape

    #print x_train, x_test, x_train.shape, x_test.shape

    model = Sequential()
    # Layer1: FC
    model.add(Dense(output_dim = 1024, input_dim = 4096, activation = 'relu'))
    model.add(Dropout(0.2))
    # Layer2: FC + Softmax
    model.add(Dense(output_dim = 2, activation = 'softmax'))
    sgd = SGD(lr = 0.005, decay = 1e-6, momentum=0.9,nesterov=True)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
    checkpointer = ModelCheckpoint(filepath = 'weights.hdf5', verbose=1, save_best_only=True)

    # Training the model
    model.fit(x_train, y_train, batch_size = 200, nb_epoch = 300, validation_split = 0.2, verbose = 1, callbacks=[checkpointer])

    testPredict = model.predict(x_test)

    yPredict = np.zeros((testPredict.shape[0],))

    for i in xrange(testPredict.shape[0]):
        if testPredict[i,0] > testPredict[i,1]:
            yPredict[i] = 0
        else:
            yPredict[i] = 1

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

    num_classes = len(feat)
    x_train, y_train, x_test, y_test = DataPreprocess(feat, num_classes)
    #print x_train.shape, x_test.shape, np.count_nonzero(y_train==0), np.count_nonzero(y_test==0)
    yPredict = Classify(x_train, y_train, x_test, y_test)
    precision = np.count_nonzero(yPredict==y_test)/float(y_test.shape[0])
    print 'Precision:', precision


    
    
