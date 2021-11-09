import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def build_linear_regressors(X,Y, test_size = 0.2, random_state = 42):


    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(X, 
                                                        Y,
                                                        test_size = test_size,
                                                        random_state = random_state)

    # creating an object of LinearRegression class
    LR   = LinearRegression() # LR is the regressor for: dataset vars -> predicted vars
    LR_2 = LinearRegression() # LR_2 is the regressor for: fitted equation -> tendency line
    # fitting the training data
    LR.fit(x_train,y_train)

    # y_prediction = LR.predict(x_test)   
    y_prediction_train = LR.predict(x_train)
    # y_verify = np.concatenate((y_test,y_prediction), axis = 1)

    LR_2.fit(y_train,y_prediction_train)

    return LR , LR_2 , (x_train, x_test, y_train, y_test)

