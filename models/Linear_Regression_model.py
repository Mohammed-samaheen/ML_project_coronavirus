import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression


class Linear_Regression:
    def __init__(self, split_data):
        self.split_data = split_data

        X_train, X_test, y_train, y_test = split_data

        y_train = np.log(y_train)
        y_test = np.log(y_test)

        self.lm = LinearRegression()
        self.lm.fit(X_train, y_train)

        self.predictions = self.lm.predict(X_test)
        self.predictions

        plt.scatter(X_test, y_test, color='red', label='test data')
        plt.scatter(X_train, y_train, color='blue', label='train data')
        plt.plot(X_test, self.predictions, linewidth=3, color="green", label='predictions')
        plt.xlabel(X_train.columns[0])
        plt.ylabel(y_train.columns[0])
        plt.legend()
        plt.show()

        sns.distplot((y_test - self.predictions))
        plt.title('univariate distribution for ' + y_train.columns[0])
        plt.show()

        self.MAE = metrics.mean_absolute_error(y_test, self.predictions)
        self.MSE = metrics.mean_squared_error(y_test, self.predictions)

        df = pd.DataFrame([self.MAE, self.MSE],
                          ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
        print('{}\n{}\n'.format(y_train.columns[0], df))

    def predict(self, value):
        return np.exp(self.lm.predict([[value]]))
