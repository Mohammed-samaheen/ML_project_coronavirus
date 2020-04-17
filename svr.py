import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# Palestine Polytechnic university (PPU)  Machine Learning course project
# Authors: 1. Ameer Takrouri
#          2. Wisam Alhroub
#          3. Mohammed Samaheen
# This university project is concerned with the study of
# predicting the number of infected people and the number of deaths of the coronavirus. By Using the
# following models: Linear regression with log values, Artificial Neutral Network (MLP Regressor)
# and Support Vector Regression (SVR)

# Support Vector Regression (SVR):
class Support_Vector_Regressor:
    def __init__(self, split_data):
        self.split_data = split_data

        X_train, X_test, y_train, y_test = split_data

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        self.lm = SVR(kernel='poly', C=100, degree=4, epsilon=.1,
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
        
        plt.plot(sx, sr, linewidth=3, color="green", label='predictions')
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


data = pd.read_csv("FINALspain.csv")

confirmed_data = train_test_split(data[["Date"]], data[["ConfirmedCom"]], test_size=0.25)
deaths_data = train_test_split(data[["Date"]][7:], data[["DeathsCom"]][7:], test_size=0.25)

Confirmed = Support_Vector_Regressor(confirmed_data)
Deaths = Support_Vector_Regressor(deaths_data)
