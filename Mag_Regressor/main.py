import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from scatter_variables import scatter_variables
from build_models import build_linear_regressors

cwd = os.getcwd()
print(cwd)

# datasets = ["Data_Cm", "Data_AMR"]
datasets = ["Data_Cm"]

LUT = pd.read_excel("LUT.xlsx" , header = 0, index_col=0)

for dataset in datasets:

    dataframe = pd.read_excel("%s.xlsx" % dataset , header = 0, index_col=0,  dtype = float)

    pred_var_count = int(LUT.loc[dataset])

    predictive_variables  = list(dataframe.keys())[:-pred_var_count]
    predicted_variables   = list(dataframe.keys())[-pred_var_count:]

    # print(predictive_variables)
    # print(predicted_variables)

    # Plot variables scatter plots
    save_flag = True
    scatter_path = os.path.join(cwd,"scatter_%s" % dataset)

    scatter_variables(dataframe,
                      predictive_variables,
                      predicted_variables,
                      save_path = scatter_path, 
                      save_flag = True)

    X = dataframe.drop(predicted_variables, axis = 1)
    Y= dataframe.drop(predictive_variables,axis = 1)

    fitted_equation, fitted_tendency, split = build_linear_regressors(X,Y)

    # Check Coeficients
    # coefs_m = fitted_equation.coef_
    # intercept_m = fitted_equation.intercept_
    # coefs_b = fitted_tendency.coef_
    # intercept_b = fitted_tendency.intercept_
    # print(intercept_m,coefs_m)
    # print(intercept_b,coefs_b)

    # Check Metrics
    x_train, x_test, y_train, y_test = split
    y_prediction_1 = fitted_equation.predict(x_test)
    y_prediction_2 = fitted_tendency.predict(y_test)

