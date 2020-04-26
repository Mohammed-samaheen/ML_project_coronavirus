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



deaths_data = train_test_split(data[["Date"]][7:].values, data[["DeathsCom"]][7:].values, test_size=0.25)

#Confirmed = MLP(confirmed_data)
#Deaths = MLP(deaths_data) to_numpy()


split_data = deaths_data

fig, subaxes = plt.subplots(4, 4, figsize=(20,20))

X_train, X_test, y_train, y_test = confirmed_data

#mlpreg = MLPRegressor (solver='lbfgs', alpha=0.01)

mae = []
mse = []

print ('{ ',172541 ,' }')

for axisRow, activationFunction, in zip (subaxes, ['tanh', 'relu','logistic', 'identity']):
    for alphas, axis in zip ([0.0001, 0.1, 1.0, 10], axisRow):
        for hid in ([2,3]):
            mlpreg = MLPRegressor (hidden_layer_sizes = [hid,1],
                                    activation = activationFunction,
                                    alpha = alphas,
                                    solver = 'lbfgs')
     
            mlpreg.fit (X_train, y_train.ravel())
            y_predict = mlpreg.predict(X_test)
            if (hid == 2):
                axis.plot(X_test, y_predict, '*')
            else :
                 axis.plot(X_test, y_predict, '^')
            axis.plot(X_test, y_test, 'o')
            axis.set_xlabel('Input feature')
            axis.set_ylabel('Target value')
            axis.set_title( "al={}, act={}, hid={}".format(alphas, activationFunction, hid) )
          
            dx = np.array([50], int)
            tempy =  mlpreg.predict(dx.reshape(1, -1))
            
            print (tempy, " ", tempy - 172541, " ", 100 - abs(1-(tempy / 172541 )) )
             
            MAE = metrics.mean_absolute_error(y_test.ravel(), y_predict)
            MSE = metrics.mean_squared_error(y_test.ravel(), y_predict)
            
            mae.append(MAE/np.mean(y_test))
            mse.append(MSE)
    
 #       plt.tight_layout()

#print (mae)
#print (mse)