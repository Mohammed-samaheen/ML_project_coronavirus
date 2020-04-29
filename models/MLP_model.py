# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:07:05 2020

@author: Amir Hashem Altakroori

This file contains the Multi-Layer Perceptron Regressor (MLP Regressor) Class

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


class MLP_Regression:
    def __init__(self, countryData, worldData = None, allDetails=False):
        
        countryData = [x.values for x in countryData]
        
        self.__dateCountry = np.concatenate((countryData[0], countryData[1]))
        self.__valueCountry = np.concatenate((countryData[2], countryData[3]), axis=None)
        
        self.worldData = worldData
        if  self.worldData is not None:
            self.worldData = [x.values for x in  self.worldData]
            self.__worldModel = self.__bestFit ( self.worldData)    
        
            
        self.__countryModel =  self.__bestFit (countryData)           
        self.__worldModel = None
            
            
    def __bestFit (self, dataSplit):
        X_train, X_test, y_train, y_test = dataSplit
        mlpreg = MLPRegressor (hidden_layer_sizes = [7],
                               alpha = 0.0001,
                               activation = 'relu',
                               solver = 'lbfgs')
        
        X_train = np.concatenate((X_train, X_test))
        y_train = np.concatenate((y_train, y_test), axis=None)
        mlpreg.fit (X_train, y_train)
        return mlpreg
        
        
    def bestPredect (self, X_test, plot=False):
        predict = self.__countryModel.predict (X_test)
        if self.worldData is not None:
            predicitWorld = self.__worldModel.predict (X_test)
            predict = self.__getApproximation (countryData = predict, worldData = predicitWorld)
            
        predict = [abs(int(x)) for x in predict]
        if plot:
            self.__carve (X_test, predict)
            
        return predict
        
    def __getApproximation (self, countryData, worldData, countryFactor = 0.80, worldFactor = 0.20):
        
        factoredCountry = [x * countryFactor for x in countryData]
        factoredWorld = [x * worldFactor for x in worldData]

        return [int(a + b) for (a, b) in zip (factoredCountry, factoredWorld)]
          
    def __carve(self, X_test, prediction):
        xList = np.arange(1, 61, 1).reshape(-1, 1)
        yList = self.bestPredect (xList)
        plt.scatter(X_test, prediction, color='red', label='test data')
        plt.scatter(self.__dateCountry, self.__valueCountry, color='blue', label='train data')
        plt.plot(xList, yList, linewidth=3, color="black", label='predictions')
        plt.legend()
        plt.show()
            