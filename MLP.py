# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:07:05 2020

@author: AnA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


# Linear regression with log values
class MLP:
    def __init__(self, split_data):
        self.split_data = split_data

  
        self.fig, self.subaxes = plt.subplots(4, 4, figsize=(11,8))
        
        X_train, X_test, y_train, y_test = self.split_data
        
        self.mlpreg = MLPRegressor (solver='lbfgs', alpha=0.01)
        self.mlpreg.fit (X_train[0], y_train[0])
        y_predict = self.mlpreg.predict(self.mlpreg.predict([X_test]))
        for axisRow, activationFunction, in zip (self.subaxes, ['tanh', 'relu','leaky_relu', 'sigmoid']):
            for alphas, axis in zip ([0.0001, 0.001, 1.0, 10], axisRow):
                self.mlpreg = MLPRegressor (hidden_layer_sizes = [100, 100],
                                            activation = activationFunction,
                                            alpha = alphas,
                                            solver = 'lbfgs')
     
         
 #               #axis.set_xlim
  #              axis.plot(X_test, y_predict, '^', markersize = 10)
   #             axis.plot(X_train, y_train, 'o')
    #            axis.set_xlabel('Input feature')
     #           axis.set_ylabel('Target value')
      #          axis.set_title('alpha={}, activation={}').format(alphas, activationFunction)
       #         plt.tight_layout()

      

data = pd.read_csv("FINALspain.csv")

confirmed_data = train_test_split(data[["Date"]].values, data[["ConfirmedCom"]].values, test_size=0.25)
deaths_data = train_test_split(data[["Date"]][7:], data[["DeathsCom"]][7:], test_size=0.25)

Confirmed = MLP(confirmed_data)
#Deaths = MLP(deaths_data) to_numpy()