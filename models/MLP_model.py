# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:07:05 2020

@author: Amir Hashem Altakroori
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


class MLP_Regression:
    def __init__(self, countryData, worldData = None, allDetails=False):
        
        countryData = [x.values for x in countryData]
        
        self.dateCountry = np.concatenate((countryData[0], countryData[1]))
        self.valueCountry = np.concatenate((countryData[2], countryData[3]), axis=None)
        
        self.usingWorld = False
        if worldData is not None:
            worldData = [x.values for x in worldData]
            self.usingWorld = True    
        
            
        self.countryModel =  self.bestFit (countryData)           
        self.worldModel = None
        if worldData is not None:
            self.worldModel = self.bestFit (worldData)
            
    def bestFit (self, dataSplit):
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
        predict = self.countryModel.predict (X_test)
        if self.usingWorld:
            predicitWorld = self.worldModel.predict (X_test)
            predict = self.getApproximation (countryData = predict, worldData = predicitWorld)
            
        predict = [abs(int(x)) for x in predict]
        if plot:
            self.carve (X_test, predict)
            
        return predict
        
    def getApproximation (self, countryData, worldData, countryFactor = 0.80, worldFactor = 0.20):
        
        factoredCountry = [x * countryFactor for x in countryData]
        factoredWorld = [x * worldFactor for x in worldData]

        return [int(a + b) for (a, b) in zip (factoredCountry, factoredWorld)]
          
    def carve(self, X_test, prediction):
        xList = np.arange(1, 61, 1).reshape(-1, 1)
        yList = self.bestPredect (xList)
        plt.scatter(X_test, prediction, color='red', label='test data')
        plt.scatter(self.dateCountry, self.valueCountry, color='blue', label='train data')
        plt.plot(xList, yList, linewidth=3, color="black", label='predictions')
        plt.legend()
        plt.show()
            