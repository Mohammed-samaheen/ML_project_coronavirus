# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:07:05 2020

@author: Amir Hashem Altakroori

This file contains the Multi-Layer Perception Regressor (MLP Regressor) Class

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class MLP_Regression:
    """
    A class used to make prediction to spain data using Multy layer multilayer perceptron (MLP)

    ...

    Attributes
    ----------
    __dateCountry : numpy array
        an array of date from the corona started in spain data
    __valueCountry : numpy array
        an array of the value of the corona results spain data
    __countryModel : M:PR model
        a model of MLPR class for the main data
     __worldModel : M:PR model
        a model of MLPR class for the secondary data
    worldData : numpy array
        an array of the secondary country value

    Methods
    -------
    __best_fit (self, dataSplit, actv = 'tanh')
        Fit the data in a MLPR model after spliting data
    best_predect (self, X_test, plot=False)
        Predict the expected vaue of the given data
    __get_approximation (self, countryData, worldData, countryFactor = 0.50, worldFactor = 0.50)
        Find the value combination with wight from two dataset values
    __carve(self, X_test, prediction)
        Plot the given data          
    """

    def __init__(self, countryData, worldData=None, allDetails=False, itr_num=100):
        """
        Parameters
        ----------
        countryData : numpay array
            Main country dataset
        worldData : numpay array, optional
            Secondary country dataset
        allDetails : boolean, optional
            Show all models details
        """

        countryData = [x.values for x in countryData]
        self.itr_num = itr_num
        ''' Use test data in prediction '''
        self.__dateCountry = np.concatenate((countryData[0], countryData[1]))
        self.__valueCountry = np.concatenate((countryData[2], countryData[3]), axis=None)

        ''' fitting data in main country model '''
        self.__countryModel = self.__best_fit(countryData)

        ''' process second dataset if it is exist '''
        self.__worldModel = None
        self.worldData = worldData

        if self.worldData is not None:
            self.worldData = [x.values for x in self.worldData]

            ''' taking log for the y-axis '''
            self.worldData[2] = np.log(self.worldData[2])
            self.worldData[3] = np.log(self.worldData[3])

            ''' fitting data in second country model '''
            self.__worldModel = self.__best_fit(self.worldData, "tanh")

    @ignore_warnings(category=ConvergenceWarning)
    def __best_fit(self, dataSplit, actv='tanh'):
        X_train, X_test, y_train, y_test = dataSplit
        mlpreg = MLPRegressor(hidden_layer_sizes=[7],
                              alpha=0.0001,
                              activation=actv,
                              solver='lbfgs', max_iter=self.itr_num)

        X_train = np.concatenate((X_train, X_test))
        y_train = np.concatenate((y_train, y_test), axis=None)
        mlpreg.fit(X_train, y_train.ravel())
        return mlpreg

    def best_predect(self, X_test, plot=False):
        """Predict the expected vaue of the given data.

        If the argument `plot` isn't passed in, the default plotting
        comand is used.

        Parameters
        ----------
        X_test : numpy array
            Date the needs to be predict
        plot : boolean, optional
            Should plot the data (default is False)
        """

        predict = self.__countryModel.predict(X_test)

        ''' When there is second dataset it needs to be combined with the main dataset'''
        if self.worldData is not None:
            predicitWorld = self.__worldModel.predict(X_test)
            predict = self.__get_approximation(countryData=predict, worldData=predicitWorld)

        ''' Get positive integer prediction '''
        predict = [abs(int(x)) for x in predict]

        ''' Plotting if asked '''
        if plot:
            self.__carve(X_test, predict)

        return predict

    def __get_approximation(self, countryData, worldData, countryFactor=0.80, worldFactor=0.20):
        """Find the value combination with wight from two dataset values.

        Parameters
        ----------
        countryData : numpy array
            main data
        worldData : numpy array
            secondary data
        countryFactor : float, optional
            wight of the main data in the summation
        worldFactor : float, optional
            wight of the secondary data in the summation
        """

        ''' Secondary dataset is predicted in log so it needs to be in the origin form '''
        worldData = np.exp(worldData)

        factoredCountry = [x * countryFactor for x in countryData]
        factoredWorld = [x * worldFactor for x in worldData]

        factoredCountry = countryFactor * countryData
        factoredWorld = worldData * worldFactor

        return [int(a + b) for (a, b) in zip(factoredCountry, factoredWorld)]

    def __carve(self, X_test, prediction):
        """ Plot the given data, with the origin data.

        Parameters
        ----------
        X_test : numpy array
            x axis value
        prediction : numpy array
            y axis value
        """

        ''' List of consecutive numbers '''
        xList = np.arange(1, 82, 1).reshape(-1, 1)

        ''' Predict value from the first day to spacife date '''
        yList = self.best_predect(xList)

        ''' Ploting requirments '''
        plt.scatter(X_test, prediction, color='red', label='test data')
        plt.scatter(self.__dateCountry, self.__valueCountry, color='blue', label='train data')
        plt.plot(xList, yList, linewidth=3, color="black", label='predictions')
        plt.legend()
        plt.show()
