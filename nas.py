import time
import json
import logging as log
import sys

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

from math import sqrt
# import keras
import tensorflow as tf
print(tf.__version__)

# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#import autokeras for Neural Architecture Search
from autokeras import StructuredDataRegressor

np.random.seed(0)
tf.random.set_seed(0)

class nas_fit(object):
    '''
    class for network
    '''

    def __init__(self, train_samples, label_array_train, test_samples, label_array_test,
                 verbose=2, max_trials = 15):
        '''
        Constructor
        Search for max_trail number of architectures and choose the best among them
        @param none
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.train_samples = train_samples
        self.label_array_train = label_array_train
        self.test_samples = test_samples
        self.label_array_test = label_array_test
        self.verbose = verbose
        self.max_trials = max_trails
        

    def implement_nas(self):
        '''
        train the network
        test the network
        predict and evaluate
        :return:
        '''
        print("Initializing network...")
        
        search = StructuredDataRegressor(max_trials=self.max_trials, loss='mean_absolute_error') #number of trial and errors allowed
        search.fit(x=self.train_samples, y=self.label_array_train, verbose=self.verbose) #fitting the model
        
        mae, acc = search.evaluate(self.test_samples, self.label_array_test, verbose=self.verbose)
        yhat = search.predict(self.test_samples)
        model1 = search.export_model()
        #model1.summary()
        #model1.save('model1.tf') #saving the model
        end_itr = time.time()

        

        print("training network is successfully completed, time: ", end_itr - start_itr)
        #print("Accuracy of the model is: ", acc)
        #return yhat , model1
        return mae , acc #it returns the mean absolute error and accuracy
