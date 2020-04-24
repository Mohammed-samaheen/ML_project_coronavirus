import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(test_size=0.10):
    file = pd.read_csv('./Data/sConfirmed.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['sConfirmed']],
                                                        test_size=test_size)
    verification = (file[['testDate']][:11], file[['testConfirmed']][:11])
    final_data = {'spainCon': ((x_train, x_test, y_train, y_test), verification)}

    file = pd.read_csv('./Data/sDeaths.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['sDeaths']],
                                                        test_size=test_size)
    verification = (file[['testDate']][:10], file[['testDeaths']][:10])
    final_data['spainDea'] = ((x_train, x_test, y_train, y_test), verification)

    file = pd.read_csv('./Data/wConfirmed.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['wConfirmed']],
                                                        test_size=test_size)
    final_data['worldCon'] = (x_train, x_test, y_train, y_test)

    file = pd.read_csv('./Data/wDeaths.csv')
    x_train, x_test, y_train, y_test = train_test_split(file[['Date']], file[['wDeaths']],
                                                        test_size=test_size)
    final_data['worldDea'] = (x_train, x_test, y_train, y_test)

    return final_data
