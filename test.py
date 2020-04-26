from util.ModifiedData import read_data
from models.Linear_Regression_model import Linear_Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# palestine polytechnic university(PPU)  Machine Learning course project
# authors:Ameer Takrouri,Wisam Alhroub and Mohammed Samaheen

#   This university project is concerned with the study of
# predict the number of infected people and the number of deaths of coronavirus.By Useing the
# following models : Linear regression with log values,Artificial Neutral Network (MLPRegressor)
# and  Support Vector Regression (SVR)

# Linear regression with log values

def calculate_verification(x, y, title):
    predict_verification = Confirmed.predict(np.reshape(x.to_numpy(), (x.shape[0], 1)))
    MAE = metrics.mean_absolute_error(y, predict_verification)
    MSE = metrics.mean_squared_error(y, predict_verification)

    df = pd.DataFrame([MAE, MSE],
                      ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
    print('verification of {}\n{}\n'.format(title, df))


data = read_data()

Confirmed = Linear_Regression(data['spainCon'][0], plot=False)
Deaths = Linear_Regression(data['spainDea'][0], plot=False)

wConfirmed = Linear_Regression(data['worldCon'], plot=False)
wDeaths = Linear_Regression(data['worldDea'], plot=False)

plt.plot(data['spainCon'][0][1], Confirmed.predictions, linewidth=3, color="green", label='spredictions')
plt.plot(data['worldCon'][1], wConfirmed.predictions, linewidth=3, color="red", label='wpredictions')
plt.scatter(data['spainCon'][0][0], np.log(data['spainCon'][0][2]), color='blue', label='train data')
plt.legend()
plt.show()

# verification
calculate_verification(data['spainCon'][1][0]['testDate'],
                       data['spainCon'][1][1], 'spain Confirmed')
calculate_verification(data['spainDea'][1][0]['testDate'],
                       data['spainDea'][1][1], 'spain Deaths')