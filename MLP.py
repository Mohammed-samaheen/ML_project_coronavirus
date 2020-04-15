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




data = pd.read_csv("FINALspain.csv")

confirmed_data = train_test_split(data[["Date"]].values, data[["ConfirmedCom"]].values, test_size=0.25)



deaths_data = train_test_split(data[["Date"]][7:], data[["DeathsCom"]][7:], test_size=0.25)

#Confirmed = MLP(confirmed_data)
#Deaths = MLP(deaths_data) to_numpy()


split_data = confirmed_data

fig, subaxes = plt.subplots(4, 4, figsize=(11,8))

X_train, X_test, y_train, y_test = split_data

#mlpreg = MLPRegressor (solver='lbfgs', alpha=0.01)

mae = []
mse = []

for axisRow, activationFunction, in zip (subaxes, ['tanh', 'relu','logistic', 'identity']):
    for alphas, axis in zip ([0.0001, 0.001, 1.0, 10], axisRow):
        mlpreg = MLPRegressor (hidden_layer_sizes = [100],
                                activation = activationFunction,
                                alpha = alphas,
                                solver = 'lbfgs')
 
        mlpreg.fit (X_train, y_train.ravel())
        y_predict = mlpreg.predict(X_test)
        axis.plot(X_test, y_predict, '^', markersize = 10)
        axis.plot(X_train, y_train, 'o')
        axis.set_xlabel('Input feature')
        axis.set_ylabel('Target value')
        axis.set_title( "al={}, act={}".format(alphas, activationFunction) )
      
          
        MAE = metrics.mean_absolute_error(y_test.ravel(), y_predict)
        MSE = metrics.mean_squared_error(y_test.ravel(), y_predict)
        
        mae.append(MAE/np.mean(y_test))
        mse.append(MSE)
    
 #       plt.tight_layout()

print (mae)
print (mse)