import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# palestine polytechnic university(PPU)  Machine Learning course project
# authors:Ameer Takrouri,Wisam Alhroub and Mohammed Samaheen

#   This university project is concerned with the study of
# predict the number of infected people and the number of deaths of coronavirus.By Useing the
# following models : Linear regression with log values,Artificial Neutral Network (MLPRegressor)
# and  Support Vector Regression (SVR)

# Linear regression with log values
class Linear_Regression:
    def __init__(self, split_data):
        self.split_data = split_data

        X_train, X_test, y_train, y_test = split_data
        print (X_train)

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


data = pd.read_csv("FINALspain.csv")

confirmed_data = train_test_split(data[["Date"]], np.log(data[["ConfirmedCom"]]), test_size=0.25)
deaths_data = train_test_split(data[["Date"]][7:], np.log(data[["DeathsCom"]][7:]), test_size=0.25)

Confirmed = Linear_Regression(confirmed_data)
Deaths = Linear_Regression(deaths_data)