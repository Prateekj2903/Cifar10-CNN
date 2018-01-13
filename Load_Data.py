import pickle
import os
import numpy as np
path = "C:/Users/windo/Desktop/New folder/DataSets/cifar-10-batches-py"
data = os.listdir(path)

def load_train_data():
    images = np.zeros((1, 3072))
    labels = np.zeros(1, 'int32')
    for file in data[1:6]:
        with open(path+'/'+file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            images = np.vstack((images, dict[b'data']))
            labels = np.hstack((labels, dict[b'labels']))
    images = images[1:]
    images = images.reshape(images.shape[0], 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    images = images / 255.0
    labels = labels[1:]
    
    train_data = {'xtrain':images, 'ytrain':labels}
    return train_data

def load_test_data():
    images = np.zeros((1, 3072))
    labels = np.zeros(1, 'int32')
    with open(path + '/' + data[len(data)-1], 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        images = np.vstack((images, dict[b'data']))
        labels = np.hstack((labels, dict[b'labels']))
    images = images[1:]
    images = images.reshape(images.shape[0], 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    images = images / 255.0
    labels = labels[1:]

    test_data = {'xtest':images, 'ytest':labels}
    return test_data

def get_labels():
    with open(path + '/' + data[0], 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'label_names']