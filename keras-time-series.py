#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:06:47 2017

@author: apm13

Keras deep learning on time series data ...! 

Following tutorial at 
http://machinelearningmastery.com/
time-series-prediction-with-deep-learning-in-python-with-keras/

Then I'll do some other stuff ..
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from random import shuffle
import math

import logging

resultsDir = './results/'

def k_fold_cross_validation(items, k, randomize=False):
    """
    Will return a list of training data sets and a list of 
    validation data sets that match that training set.
    
    I spent time writing this and then realised that I'm modelling a time
    series so cross-validation could be mental ...
    
    # if you want to do cross validation, for leave one out
    # set the number of folds to be equal to len(dataset)
    
    training, validation = k_fold_cross_validation(dataset, 4)
    
    # test k_fold function
    items = range(97)
    for training, validation in k_fold_cross_validation(items, 7):
        for item in items:
            assert (item in training) ^ (item in validation)
    """
    
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in xrange(k)]
    
    training = []
    validation = []
    for i in xrange(k):
        validation.append(slices[i])
        training.append([item
                    for s in slices if s is not validation
                    for item in s])
    return training, validation


def func(x):
    """
    Can use for convertion to float test of df values, not used here now ...
    """
    try:
        return float(x)
    except ValueError:
        return np.nan


def create_dataset(dataset, lookBack=1):
    """
    Convert an array of values into a dataset matrix.
    lookBack is how many previous time steps you want to use to predict next
    value.
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-lookBack-1):
         a = dataset[i:(i+lookBack)][0] # I want to append only values not a list.
         dataX.append(a)
         dataY.append(dataset[i + lookBack])
    return np.array(dataX), np.array(dataY)

def define_sequential_model(neuralNetArchitecture):
    
    model = Sequential()
    #add the first and hidden layers based on neuralNetArch.
    for i in range(len(neuralNetArchitecture)):
        if i==0:
            model.add(Dense(neuralNetArchitecture[i], input_dim=1, activation='relu'))
        else:
            model.add(Dense(neuralNetArchitecture[i]))
    # output layer        
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    logger.info("Have set up neural network model ... more logging to follow.")
    
    return model

def train_predict_plot_multilayerperceptron(dataset, split=0.75, lookBack=1):
    """
    dataset should be a numpy array of the time series values.
    
    """
    # split into train and test sets
    # remember this way of doing it is only ok because this is a time
    # series!
    trainSize = int(len(dataset)*split)
    #testSize = len(dataset) - trainSize
    train, test = dataset[0:trainSize], dataset[trainSize:len(dataset)]
    
    # reshape into X=t and Y=t+1
    #lookBack = 1
    trainX, trainY = create_dataset(train, lookBack)
    testX, testY = create_dataset(test, lookBack)
    
    # Go for a multilayer perceptron:
    #   1 input layer
    #   1 hidden layer, 8 neurons
    #   output layer.
    #
    # Fit: mean squared error.
    layersHiddenArchitecture = [8,8] # two layers of 8 neurons.
    model = define_sequential_model(layersHiddenArchitecture)
        
    model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    
    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
     
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[lookBack:len(trainPredict)+lookBack] = trainPredict[:, 0]
     
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict)+(lookBack*2)+1:len(dataset)-1] = testPredict[:, 0]
    
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig(resultsDir+'test.pdf')
    return 0
        
    

if __name__ == "__main__":
    # create logger
    logger = logging.getLogger('kerasTimeSeries')
    logger.setLevel(logging.DEBUG)
    
    # create console handler and set level to debug
    ch = logging.FileHandler('./logs/kerasTimeSeries.log')
    ch.setLevel(logging.DEBUG)
    
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # add formatter to ch
    ch.setFormatter(formatter)
    
    # add ch to logger
    logger.addHandler(ch)
    
    np.random.seed(10)
    # load data
    dfair = pd.read_csv("./data/international-airline-passengers.csv",
                        usecols=[1], engine='python', skipfooter=3)
    dataset = dfair.values.ravel()
    dataset = dataset.astype('float32')
    logger.info("Data read in")
    
    train_predict_plot_multilayerperceptron(dataset)
    logger.info("Multilayer perceptron trained, predicted and plotted... DONE!")
   