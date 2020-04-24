import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVR


# Palestine Polytechnic university (PPU)  Machine Learning course project
# Authors: 1. Ameer Takrouri
#          2. Wisam Alhroub
#          3. Mohammed Samaheen
# This university project is concerned with the study of
# predicting the number of infected people and the number of deaths of the coronavirus. By Using the
# following models: Linear regression with log values, Artificial Neutral Network (MLP Regressor)
# and Support Vector Regression (SVR)

# Support Vector Regression (SVR):

def read_data(log = False):
    file = pd.read_csv('./Data/sConfirmed.csv')
    x_train, y_train = (file['Date'], file['sConfirmed'])
    x_test, y_test = (file['testDate'][:11], file['testConfirmed'][:11])

    if log:
        y_train = np.log(y_train)
        y_test = np.log(y_test)

    final_data = {'spainCon': (x_train, x_test, y_train, y_test)}
    
    file = pd.read_csv('./Data/sDeaths.csv')
    x_train, y_train = (file['Date'], file['sDeaths'])
    x_test, y_test = (file['testDate'][:10], file['testDeaths'][:10])

    if log:
        y_train = np.log(y_train)
        y_test = np.log(y_test)

    final_data['spainDea'] = (x_train, x_test, y_train, y_test)

    file = pd.read_csv('./Data/wConfirmed.csv')
    x_train, y_train = (file['Date'], file['wConfirmed'])
    final_data['worldCon'] = (x_train, y_train)

    file = pd.read_csv('./Data/wDeaths.csv')
    x_train, y_train = (file['Date'], file['wDeaths'])
    final_data['worldDea'] = (x_train, y_train)

    return final_data

class Support_Vector_Regressor:
    def __init__(self, split_data):
        self.split_data = split_data
        
        X_train, X_test, y_train, y_test = split_data
        
        
        print("X_test is:\n", y_train)
        
        self.lm = SVR(kernel='poly', C=100, degree=2, epsilon=.1,
               coef0=4)
        self.lm.fit(X_train, y_train)
        
        print(len(self.lm.support_vectors_), len(X_test))

        self.predictions = self.lm.predict(X_test)
        

        plt.scatter(X_test.ravel(), y_test.ravel(), color='red', label='test data')
        plt.scatter(X_test.ravel(), self.predictions, color='brown', label='Predicted Values')
        '''plt.scatter(,self.lm.support_vectors_,  color='white', label='Supporting Vectors',
                    edgecolors='black')'''
        plt.scatter(X_train, y_train, color='blue', label='train data')
        
        sr = np.copy(self.predictions)
        sx = np.copy(X_test.ravel())
        sr = np.sort(sr)
        sx = np.sort(sx)
        
        plt.plot(sx, sr, linewidth = 3, color="green", label='predictions')
        plt.xlabel(X_train.columns[0])
        plt.ylabel(y_train.columns[0])
        plt.legend()
        plt.show()

        sns.distplot((y_test.ravel() - self.predictions))
        plt.title('univariate distribution for ' + y_train.columns[0])
        plt.show()

        self.MAE = metrics.mean_absolute_error(y_test.ravel(), self.predictions)
        self.MSE = metrics.mean_squared_error(y_test.ravel(), self.predictions)

        df = pd.DataFrame([self.MAE, self.MSE],
                          ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
        print('{}\n{}\n'.format(y_train.columns[0], df))

    def predict(self, value):
        return np.exp(self.lm.predict([[value]]))

data = read_data()

Confirmed = Support_Vector_Regressor(data['spainCon'])
Deaths = Support_Vector_Regressor(data['spainDea'])
