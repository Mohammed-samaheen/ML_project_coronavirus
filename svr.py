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

# Support Vector Regression (SVR) By Wisam Alhroub:

def read_data(test_size=0.10):
    file = pd.read_csv('./Data/sConfirmed.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['sConfirmed']],
                                                        test_size=test_size)
    verification = (file[['testDate']][:11], file[['testConfirmed']][:11])
    final_data = {'spainCon': ((x_train, x_test, y_train, y_test), verification)}

    file = pd.read_csv('./Data/sDeaths.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['sDeaths']],
                                                        test_size=test_size)
    verification = (file[['testDate']][:10], file[['testDeaths']][:10])
    final_data['spainDea'] = ((x_train, x_test, y_train, y_test), verification)

    file = pd.read_csv('./Data/wConfirmed.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['wConfirmed']],
                                                        test_size=test_size)
    final_data['worldCon'] = (x_train, x_test, y_train, y_test)

    file = pd.read_csv('./Data/wDeaths.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['wDeaths']],
                                                        test_size=test_size)
    final_data['worldDea'] = (x_train, x_test, y_train, y_test)

    return final_data

class Support_Vector_Regressor:
    def __init__(self, split_data):
        
        self.split_data = split_data
        
        X_train, X_test, y_train, y_test = split_data
        
        self.lm = SVR(kernel= 'poly', C= 30, degree=4 ,epsilon=.01, coef0=1)
        
        #To sort the test data to show in an apprpriate way:
        X_test = pd.DataFrame.sort_index(X_test, axis=0)
        y_test = pd.DataFrame.sort_index(y_test, axis=0)
        
        self.lm.fit(X_train, y_train)

        self.predictions = self.lm.predict(X_test)
        
        plt.scatter(X_test, y_test, color='red', label='test data')
        plt.scatter(X_test, self.predictions, color='brown', label='Predicted Values')
        plt.scatter(X_train, y_train, color='blue', label='train data')
        '''plt.scatter(self.lm.support_vectors_, y_train, color='white', label='Supporting Vectors',
                    edgecolors='black')'''
        
        
        plt.plot(X_test, self.predictions, linewidth = 3, color="green", label='predictions')
        plt.xlabel(X_train.columns[0])
        plt.ylabel(y_train.columns[0])
        plt.legend()
        plt.show()
        
        sns.distplot((np.array(y_test) - self.predictions))
        plt.title('univariate distribution for ' + y_train.columns[0])
        plt.show()

        self.MAE = metrics.mean_absolute_error(y_test, self.predictions)
        self.MSE = metrics.mean_squared_error(y_test, self.predictions)

        df = pd.DataFrame([self.MAE, self.MSE],
                          ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
        print('{}\n{}\n'.format(y_train.columns[0], df))

    def predict(self, value):
        return np.exp(self.lm.predict([[value]]))

data = read_data()



Confirmed = Support_Vector_Regressor(data['spainCon'][0])
Deaths = Support_Vector_Regressor(data['spainDea'][0])
