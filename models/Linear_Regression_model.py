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
