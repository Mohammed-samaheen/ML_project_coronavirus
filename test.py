from util.ModifiedData import read_data
from models.MLP_model import MLP_Regression

from models.Linear_Regression_model import Linear_Regression, Linear_summary

import numpy as np
import pandas as pd
from sklearn import metrics


# palestine polytechnic university(PPU)  Machine Learning course project
# authors:Ameer Takrouri,Wisam Alhroub and Mohammed Samaheen

#   This university project is concerned with the study of
# predict the number of infected people and the number of deaths of coronavirus.By Useing the
# following models : Linear regression with log values,Artificial Neutral Network (MLPRegressor)
# and  Support Vector Regression (SVR)

# Linear regression with log values

def verification_MSE(x, y, obj, title):
    y = np.log(y)
    predict_verification = np.log(obj.predict(np.reshape(x.to_numpy(), (x.shape[0], 1))))
    MAE = metrics.mean_absolute_error(y, predict_verification)
    MSE = metrics.mean_squared_error(y, predict_verification)

    df = pd.DataFrame([MAE, MSE],
                      ['mean absolute error (MAE)', 'mean squared error (MSE)'], columns=['Result'])
    print('verification of {}\n{}\n'.format(title, df))




data = read_data(0.2)

mlpSpainConfirmed = MLP_Regression(data['spainCon'][0], allDetails=False)
mlpConfirmedPrediction = mlpSpainConfirmed.bestPredect(((data['spainCon'][1][0]['testDate']).values).reshape(-1, 1), plot=True)

confirmedDiference = [(a-b) for a, b in zip (mlpConfirmedPrediction, data['spainCon'][1][1].values)]
print(confirmedDiference)
print ( metrics.mean_absolute_error(mlpConfirmedPrediction, data['spainCon'][1][1]) )

mlpSpainDeath = MLP_Regression(data['spainDea'][0], allDetails=False)
mlpDeathPrediction = mlpSpainDeath.bestPredect(((data['spainDea'][1][0]['testDate']).values).reshape(-1, 1), plot=True)
deathDiference = [(a-b) for a, b in zip (mlpDeathPrediction, data['spainCon'][1][1].values)]
print(deathDiference)
print ( metrics.mean_absolute_error(mlpDeathPrediction, data['spainDea'][1][1]) )


Confirmed = Linear_Regression(data['spainCon'][0], plot=False)
Deaths = Linear_Regression(data['spainDea'][0], plot=False)

spain_confirmed = Linear_Regression(data['spainCon'][0], plot=False)
spain_deaths = Linear_Regression(data['spainDea'][0], plot=False)


world_confirmed = Linear_Regression(data['worldCon'], plot=False)
world_deaths = Linear_Regression(data['worldDea'], plot=False)

# summary plot of confirmed and deaths in spain and the world
summary = Linear_summary(data)
summary.confirmed_summary(spain_confirmed, world_confirmed)
summary.deaths_summary(spain_deaths, world_deaths)

# test the verification data
verification_MSE(data['spainCon'][1][0]['testDate'],
                 data['spainCon'][1][1], spain_confirmed, 'spain Confirmed')
verification_MSE(data['spainDea'][1][0]['testDate'],
                 data['spainDea'][1][1], spain_deaths, 'spain Deaths')

# Predict the future value of confirmed and deaths
print('On the {} day the number of confirmed is {}'.format(
    60,(spain_confirmed.predict([[60]])-world_confirmed.predict([[60]]))/2
))
print('On the {} day the number of deaths is {}'.format(
    60,(spain_deaths.predict([[60]])-world_deaths.predict([[60]]))/2
))


# verification
calculate_verification(data['spainCon'][1][0]['testDate'],
                       data['spainCon'][1][1], 'spain Confirmed')
calculate_verification(data['spainDea'][1][0]['testDate'],
                       data['spainDea'][1][1], 'spain Deaths')


