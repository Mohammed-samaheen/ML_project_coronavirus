# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:07:05 2020

@author: AnA
"""

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


class MLP_Regression:
    def __init__(self, spainData, worldData, plot=False, allDetails=False):
        
        spainData = [x.values for x in spainData]
        self.y_test = spainData[3]
        worldData = [y.values for y in worldData]
        
        self.predictionSpain = self.bestPredect(spainData)
        self.predictionWorld = self.bestPredect(worldData)
        
        self.standardizedData()
        
        print(metrics.mean_absolute_error(self.y_test.ravel(), self.predictionSpain))
        
        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            self.prediction = self.getApproximation (spainFactor = x, worldFactor = (1 - x))
            print(x *100,"% spain: ", metrics.mean_absolute_error(self.y_test.ravel(), self.prediction))                    
        
    def getApproximation (self, spainFactor = 0.75, worldFactor = 0.25):
        
        factoredSpain = [x * spainFactor for x in self.predictionSpain]
        factoredWorld = [x * worldFactor for x in self.predictionWorld]

        return [int(a + b) for (a, b) in zip (factoredSpain, factoredWorld)]
        
    def standardizedData (self):
        
        minLin = min (len(self.predictionSpain), len(self.predictionWorld))
       
        if (len(self.predictionSpain) > len(self.predictionWorld)):
            self.predictionWorld = self.predictionWorld + self.predictionSpain[minLin:]
        else:
            self.predictionSpain = self.predictionSpain [ : minLin]
            self.predictionWorld = self.predictionWorld [ : minLin]
            
        
    def bestPredect (self, dataSplit):
        
        X_train, X_test, y_train, y_test = dataSplit
        
        mlpreg = MLPRegressor (hidden_layer_sizes = [1000],
                               alpha = 0.0001,
                               solver = 'lbfgs')
     
        mlpreg.fit (X_train, y_train.ravel())
        
        y_predict = mlpreg.predict(X_test)
        y_predict = [abs (int(x)) for x in y_predict]
        
        return y_predict
            
    def carve(self):
        plt.scatter(self.X_test, self.y_predict, color='red', label='test data')
        plt.scatter(self.X_train, self.y_train, color='blue', label='train data')
        plt.plot(self.X_test, self.y_predict, linewidth=3, color="green", label='predictions')
        plt.legend()
        plt.show()
            
        
       


'''
data = pd.read_csv("FINALspain.csv")

confirmed_data = train_test_split(data[["Date"]].values, data[["ConfirmedCom"]].values, test_size=0.25)



deaths_data = train_test_split(data[["Date"]][7:].values, data[["DeathsCom"]][7:].values, test_size=0.25)

#Confirmed = MLP(confirmed_data)
#Deaths = MLP(deaths_data) to_numpy()
'''


'''
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
 
 
 '''
