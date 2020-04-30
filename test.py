from util.ModifiedData import read_data
from models.MLP_model import MLP_Regression
from models.Support_Vector_Regression_model import Support_Vector_Regressor
from models.Linear_Regression_model import Linear_Regression, Linear_summary
from sklearn import metrics

# palestine polytechnic university(PPU)  Machine Learning course project
# authors: Ameer Takrouri, Wisam Alhroub and Mohammed Samaheen

#   This university project is concerned with the study of
# predict the number of infected people and the number of deaths of coronavirus.By Useing the
# following models : Linear regression with log values,Artificial Neutral Network (MLPRegressor)
# and  Support Vector Regression (SVR)


# MLP Regressor Model:

data = read_data(0.2)

mlpSpainConfirmed = MLP_Regression(data['spainCon'][0], allDetails=False)
mlpConfirmedPrediction = mlpSpainConfirmed.bestPredect((data['spainCon'][1][0]['testDate']).values.reshape(-1, 1),
                                                       plot=True)

confirmedDiference = [(a - b) for a, b in zip(mlpConfirmedPrediction, data['spainCon'][1][1].values)]
print(confirmedDiference)
print(metrics.mean_absolute_error(mlpConfirmedPrediction, data['spainCon'][1][1]))
print("MSE: ", metrics.mean_absolute_error(mlpConfirmedPrediction, data['spainCon'][1][1]))

mlpSpainDeath = MLP_Regression(data['spainDea'][0], allDetails=False)
mlpDeathPrediction = mlpSpainDeath.bestPredect((data['spainDea'][1][0]['testDate']).values.reshape(-1, 1), plot=True)
deathDiference = [(a - b) for a, b in zip(mlpDeathPrediction, data['spainCon'][1][1].values)]
print(deathDiference)
print(metrics.mean_absolute_error(mlpDeathPrediction, data['spainDea'][1][1]))
print("MSE: ", metrics.mean_absolute_error(mlpDeathPrediction, data['spainDea'][1][1]))

Confirmed = Linear_Regression(data['spainCon'][0], plot=False)
Deaths = Linear_Regression(data['spainDea'][0], plot=False)

###################################################################################################################

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

# Predict the future value of confirmed and deaths
print('On the {} day the number of confirmed is {}'.format(
    60, (spain_confirmed.predict([[60]]) - world_confirmed.predict([[60]])) / 2
))
print('On the {} day the number of deaths is {}'.format(
    60, (spain_deaths.predict([[60]]) - world_deaths.predict([[60]])) / 2
,'\n\n'))

###################################################################################################################

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

