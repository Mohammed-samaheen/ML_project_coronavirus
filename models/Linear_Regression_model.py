
# @author:Mohammed Khaled Samaheen
# This file contains the Linear Regression Class model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression


class Linear_Regression:
    def __init__(self, split_data, plot=True):
        self.split_data = split_data

        self.X_train, self.X_test, self.y_train, self.y_test = split_data

        self.y_train = np.log(self.y_train)
        self.y_test = np.log(self.y_test)

        self.lm = LinearRegression()
        self.lm.fit(self.X_train, self.y_train)

        self.predictions = self.lm.predict(self.X_test)
        self.predictions

        if plot:
            self.carve()
            self.distribution()

        self.MAE = metrics.mean_absolute_error(self.y_test, self.predictions)
        self.MSE = metrics.mean_squared_error(self.y_test, self.predictions)

        df = pd.DataFrame([self.MAE, self.MSE],
                          ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
        print('{}\n{}\n'.format(self.y_train.columns[0], df))

    def predict(self, value):
        return np.exp(self.lm.predict(value))

    def carve(self):
        plt.scatter(self.X_test, self.y_test, color='red', label='test data')
        plt.scatter(self.X_train, self.y_train, color='blue', label='train data')
        plt.plot(self.X_test, self.predictions, linewidth=3, color="green", label='predictions')
        plt.xlabel(self.X_train.columns[0])
        plt.ylabel(self.y_train.columns[0])
        plt.legend()
        plt.show()

    def distribution(self):
        sns.distplot((self.y_test - self.predictions))
        plt.title('univariate distribution for ' + self.y_train.columns[0])
        plt.show()

    def linear_verification(self, x, y, title):
        predict_verification = self.predict(np.reshape(x.to_numpy(), (x.shape[0], 1)))
        MAE = (metrics.mean_absolute_error(y, predict_verification))
        MSE = (metrics.mean_squared_error(y, predict_verification))

        df = pd.DataFrame([MAE, MSE],
                          ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
        print('verification of {}\n{}\n'.format(title, df))

class Linear_summary:
    def __init__(self,data):
        self.data=data

    def confirmed_summary(self, Confirmed, wConfirmed):
        plt.plot(self.data['spainCon'][0][1], Confirmed.predictions, linewidth=3, color="green",
                 label='spain prediction line')
        plt.scatter(self.data['spainCon'][1][0]['testDate'],
                    np.log(self.data['spainCon'][1][1]), color='orange', marker='^', label='verification data')
        plt.scatter(self.data['spainCon'][1][0]['testDate'],
                    np.log(Confirmed.predict(np.reshape(self.data['spainCon'][1][0]['testDate'].to_numpy(), (11, 1)))),
                    color='black', marker='v', label='spain prediction point')
        plt.plot(self.data['worldCon'][1], wConfirmed.predictions, linewidth=3, color="red", label='world predictions line')
        plt.scatter(self.data['spainCon'][0][0], np.log(self.data['spainCon'][0][2]), color='blue', label='train data')
        plt.title('spain Confirmed')
        plt.legend()
        plt.show()

    def deaths_summary(self, Deaths, wDeaths):
        plt.plot(self.data['spainDea'][0][1], Deaths.predictions, linewidth=3, color="green",
                 label='spain prediction line')
        plt.scatter(self.data['spainDea'][1][0]['testDate'],
                    np.log(self.data['spainDea'][1][1]), color='orange', marker='^', label='verification data')
        plt.scatter(self.data['spainDea'][1][0]['testDate'],
                    np.log(Deaths.predict(np.reshape(self.data['spainDea'][1][0]['testDate'].to_numpy(), (9, 1)))),
                    color='black', marker='v', label='spain prediction point')
        plt.plot(self.data['worldDea'][1], wDeaths.predictions, linewidth=3, color="red", label='world predictions line')
        plt.scatter(self.data['spainDea'][0][0], np.log(self.data['spainDea'][0][2]), color='blue', label='train data')
        plt.title('spain Deaths')
        plt.legend()
        plt.show()

