from util.ModifiedData import read_data
from util.summary_model import *
from util.const import *
from models.MLP_model import MLP_Regression
from models.Support_Vector_Regression_model import Support_Vector_Regressor
from models.Linear_Regression_model import Linear_Regression, Linear_summary
from sklearn import metrics
import numpy as np

# palestine polytechnic university(PPU)  Machine Learning course project
# authors: Ameer Takrouri, Wisam Alhroub and Mohammed Samaheen

#   This university project is concerned with the study of
# predict the number of infected people and the number of deaths of coronavirus.By Useing the
# following models : Linear regression with log values,Artificial Neutral Network (MLPRegressor)
# and  Support Vector Regression (SVR)


data = read_data(0.2)
# MLP Regressor Model:

''' Confirmed model starts '''
print(OKBLUE+'Multi-Layer Perception -MLP-\n'+
      '--------'*12+ENDC)
print("\n\nConfirmed in Spain by: Multi-Layer Perceptron Regressor (MLP Regressor) Model.")

# MLP Regressor Model to predict the Confirmed in Spain
mlpSpainConfirmed = MLP_Regression(data['spainCon'][0], data['spainCon'][0], allDetails=False,itr_num=1000)
mlpConfirmedPrediction = mlpSpainConfirmed.best_predect((data['spainCon'][1][0]['testDate']).values.reshape(-1, 1),
                                                        plot=True)

# Print Mean Absolute error
print(WARNING+f"MAE: {metrics.mean_absolute_error(mlpConfirmedPrediction, data['spainCon'][1][1])}"+ENDC)

# Print Mean Squared Error
print(WARNING+f"MSE: {metrics.mean_squared_error(mlpConfirmedPrediction, data['spainCon'][1][1])}"+ENDC)

''' Confirmed model ends '''

''' Death model starts '''

print("\n\nDeath in Spain by: Multi-Layer Perceptron Regressor (MLP Regressor) Model.")

# MLP Regressor Model to predict the Death in Spain
mlpSpainDeath = MLP_Regression(data['spainDea'][0], allDetails=False,itr_num=1000)

# Print Mean Squared Error
mlpDeathPrediction = mlpSpainDeath.best_predect((data['spainDea'][1][0]['testDate']).values.reshape(-1, 1), plot=True)

# Print Mean Absolute error
print(WARNING+f"MAE:  {metrics.mean_absolute_error(mlpDeathPrediction, data['spainDea'][1][1])}"+ENDC)


# Print Mean Squared Error
print(WARNING+f"MSE:  {metrics.mean_squared_error(mlpDeathPrediction, data['spainDea'][1][1])}"+ENDC)

''' Death model ends '''

###################################################################################################################
print('\n'+OKBLUE+'Linear Regression\n' +
      '--------'*12+ENDC+'\n')
# Linear Regression model


spain_confirmed = Linear_Regression(data['spainCon'][0], plot=False)
spain_deaths = Linear_Regression(data['spainDea'][0], plot=False)

world_confirmed = Linear_Regression(data['worldCon'], plot=False)
world_deaths = Linear_Regression(data['worldDea'], plot=False)

# summary plot of confirmed and deaths in spain and the world
summary = Linear_summary(data)
summary.confirmed_summary(spain_confirmed, world_confirmed)
summary.deaths_summary(spain_deaths, world_deaths)

# test the verification data
spain_confirmed.linear_verification(data['spainCon'][1][0]['testDate'],
                                    data['spainCon'][1][1], 'spain Confirmed')
spain_deaths.linear_verification(data['spainDea'][1][0]['testDate'],
                                 data['spainDea'][1][1], 'spain Deaths')


###################################################################################################################
print('\n'+OKBLUE+'Support Vector Regression\n'+
      '--------'*12+ENDC+'\n')
# Support Vector Regression Model:

Spain_Confirmed = Support_Vector_Regressor(data['spainCon'][0])
Spain_Deaths = Support_Vector_Regressor(data['spainDea'][0])

World_Confirmed = Support_Vector_Regressor(data['worldCon'])
World_Deaths = Support_Vector_Regressor(data['worldDea'])

Spain_Confirmed.plotResults("Spain's Confirmed Cases")
Spain_Deaths.plotResults("Spain's Death Cases")
World_Confirmed.plotResults("World's Confirmed Cases")
World_Deaths.plotResults("World's Death Cases")

Spain_Confirmed.SVRVerification(data['spainCon'][1][0]['testDate'],
                                data['spainCon'][1][1], "Spain's Confirmed Cases")
Spain_Deaths.SVRVerification(data['spainDea'][1][0]['testDate'],
                             data['spainDea'][1][1], "Spain's Death Cases")

###################################################################################################################
# summary

day = int(input('\n'+OKGREEN+'enter after how many days from the last study :'+ENDC))
result = predict_summary(day,
                         mlp_regression=(mlpSpainConfirmed, mlpSpainDeath),
                         linear_regression=((spain_confirmed, spain_deaths),
                                            (world_confirmed, world_deaths)),
                         support_vector_regressor=(Spain_Confirmed, Spain_Deaths))
print(f"\n{result}")


