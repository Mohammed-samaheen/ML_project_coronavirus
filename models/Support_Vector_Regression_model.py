'''
    Auther: Wisam Alhroub
    This file consists of the Support Vector Regression Model
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVR


#########################################################################################################

class Support_Vector_Regressor:

    def __init__(self, split_data, verify=False):
        self.split_data = split_data
        self.X_train, self.X_test, self.y_train, self.y_test = split_data

        self.lm = SVR(kernel='poly', C=150, degree=3, epsilon=.001, coef0=5)

        # To sort the test data to show in an apprpriate way:
        self.X_test = pd.DataFrame.sort_index(self.X_test, axis=0)
        self.y_test = pd.DataFrame.sort_index(self.y_test, axis=0)

        self.lm.fit(self.X_train.values, self.y_train.values.ravel())

        self.predictions = self.lm.predict(self.X_test)

    '''---------------------------------------------'''

    def predictValues(self, day):
        return self.lm.predict(day)

    '''---------------------------------------------'''

    def plotResults(self, title, showDist=False):
        plt.scatter(self.X_test, self.y_test, color='red', label='test data')
        plt.scatter(self.X_test, self.predictions, color='brown', marker='^', label='Predicted Values')
        plt.scatter(self.X_train, self.y_train, color='blue', label='train data')
        plt.plot(self.X_test, self.predictions, linewidth=3, color="green", label='predictions')
        plt.title(title)
        plt.xlabel(self.X_train.columns[0])
        plt.ylabel(self.y_train.columns[0])
        plt.legend()
        plt.show()

        if showDist:
            sns.distplot((np.array(self.y_test) - self.predictions))
            plt.title('Univariate distribution for ' + self.y_train.columns[0])
            plt.show()

    '''---------------------------------------------'''

    def SVRVerification(self, x, y, title):
        verification = self.predictValues(np.reshape(x.to_numpy(), (x.shape[0], 1)))
        self.MAE = (metrics.mean_absolute_error(y, verification))
        self.MSE = (metrics.mean_squared_error(y, verification))

        MSAE = pd.DataFrame([self.MAE, self.MSE],
                            ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
        print('Support Vector Regressor Verification of', title, ':\n', MSAE)

    '''---------------------------------------------'''

#########################################################################################################
