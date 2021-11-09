import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from plotting_util import scatter_variables, plot_fitted_equation
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
    Y = dataframe.drop(predictive_variables,axis = 1)

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(X, 
                                                        Y,
                                                        test_size = 0.2,
                                                        random_state = 42)


    fitted_equation, fitted_tendency = build_linear_regressors(x_train,
                                                               x_test,
                                                               y_train,
                                                               y_test)

    # Check Coeficients
    # coefs_m = fitted_equation.coef_
    # intercept_m = fitted_equation.intercept_
    # coefs_b = fitted_tendency.coef_
    # intercept_b = fitted_tendency.intercept_
    # print(intercept_m,coefs_m)
    # print(intercept_b,coefs_b)

    # Check Metrics
    y_prediction_1 = fitted_equation.predict(x_test)
    y_prediction_2 = fitted_tendency.predict(y_test)
    
    # Predicting the accuracy score
    # Fitted Equation:
    FE_score = r2_score(y_test,y_prediction_1)
    print("Fitted Equation:")
    print('r2 score for the fitted equation is: %f'% FE_score)
    print('mean_sqrd_error is: ' , mean_squared_error(y_test,y_prediction_1))
    print('root_mean_squared error is: ',np.sqrt(mean_squared_error(y_test,y_prediction_1)))
    print("")

    # Fitted Tendency:
    FT_score = r2_score(y_test,y_prediction_2)
    print("Fitted Tendency Line:")
    print('r2 score for the fitted tendency line is: %f'% FT_score)
    print('mean_sqrd_error is: ' , mean_squared_error(y_test,y_prediction_2))
    print('root_mean_squared error is: ',np.sqrt(mean_squared_error(y_test,y_prediction_2)))
    print("")
    
    # Plot Fitted Equation


    # Save data