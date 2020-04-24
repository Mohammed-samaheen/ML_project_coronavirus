from util.ModifiedData import read_data
from models.Linear_Regression_model import Linear_Regression

# palestine polytechnic university(PPU)  Machine Learning course project
# authors:Ameer Takrouri,Wisam Alhroub and Mohammed Samaheen

#   This university project is concerned with the study of
# predict the number of infected people and the number of deaths of coronavirus.By Useing the
# following models : Linear regression with log values,Artificial Neutral Network (MLPRegressor)
# and  Support Vector Regression (SVR)

# Linear regression with log values


data = read_data()

Confirmed = Linear_Regression(data['spainCon'][0])
Deaths = Linear_Regression(data['spainDea'][0])

wConfirmed=Linear_Regression(data['worldCon'])
wDeaths = Linear_Regression(data['worldDea'])
