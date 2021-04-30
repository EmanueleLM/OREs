import numpy as np
import keras
import tensorflow as tf
import string
from pandas import read_csv
from tensorflow.keras.datasets import imdb
from keras.datasets import mnist
from keras import backend as K

def load_SST(num_samples=-1, return_text=False):
    # Load STT dataset (eliminate punctuation, add padding etc.)
    X_train = read_csv('./data/SST_2/training/SST_2__FULL.csv', sep=',',header=None).values
    X_test = read_csv('./data/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
    y_train, y_test = [], []
    for i in range(len(X_train)):
        r, s = X_train[i]  # review, score (comma separated in the original file)
        X_train[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_train.append((0 if s.strip()=='negative' else 1))
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_test.append((0 if s.strip()=='negative' else 1))
    X_train, X_test = X_train[:,0], X_test[:,0]
    n = num_samples  # you may want to take just some samples (-1 to take them all)
    X_train = X_train[:n]
    X_test = X_test[:n]
    y_train = y_train[:n]
    y_test = y_test[:n]
    if return_text is False:
        raise(NotImplementedError("Dataset can't be loaded in this way and must be splitted, check train_STT.py for an example"))
    else:
        return (X_train, y_train), (X_test, y_test)

def load_MNIST(num_samples=-1):
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)