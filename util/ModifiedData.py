import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

last_day_of_confirmed = 0
last_day_of_deaths = 0


def read_data(test_size=0.10):
    global last_day_of_confirmed, last_day_of_deaths

    # sConfirmed.csv
    file = pd.read_csv('./Data/sConfirmed.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['sConfirmed']],
                                                        test_size=test_size)
    verification = (file[['testDate']][:11], file[['testConfirmed']][:11])
    final_data = {'spainCon': ((x_train, x_test, y_train, y_test), verification)}
    last_day_of_confirmed = int(np.max([file[['Date']].max(), file[['testDate']][:11].max()]))

    # sDeaths.csv
    file = pd.read_csv('./Data/sDeaths.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['sDeaths']],
                                                        test_size=test_size)
    verification = (file[['testDate']][:9], file[['testDeaths']][:9])
    final_data['spainDea'] = ((x_train, x_test, y_train, y_test), verification)
    last_day_of_deaths = int(np.max([file[['testDate']][:9].max(), file[['Date']].max()]))

    # wConfirmed.csv
    file = pd.read_csv('./Data/wConfirmed.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['wConfirmed']],
                                                        test_size=test_size)
    final_data['worldCon'] = (x_train, x_test, y_train, y_test)

    # wDeaths.csv
    file = pd.read_csv('./Data/wDeaths.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['wDeaths']],
                                                        test_size=test_size)
    final_data['worldDea'] = (x_train, x_test, y_train, y_test)

    return final_data


def max_day():
    return last_day_of_confirmed, last_day_of_deaths
