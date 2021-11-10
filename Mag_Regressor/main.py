import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from plotting_util import scatter_variables, plot_fitted_equation
from build_models import build_linear_regressor

cwd = os.getcwd()
print(cwd)

datasets = ["Data_Cm", "Data_AMR"]
# datasets = ["Data_Cm"]
# datasets = ["Data_AMR"]

LUT = pd.read_excel("LUT.xlsx" , header = 0, index_col=0)

for dataset in datasets:

    print(dataset)

    dataframe = pd.read_excel("%s.xlsx" % dataset , header = 0, index_col=0,  dtype = float)

    pred_var_count = int(LUT.loc[dataset])

    predictive_variables  = list(dataframe.keys())[:-pred_var_count]
    predicted_variables   = list(dataframe.keys())[-pred_var_count:]

    # print(predictive_variables)
    # print(predicted_variables)

    # Plot variables scatter plots
    save_flag = True
    scatter_path = os.path.join(cwd,"scatter_%s" % dataset)

    print("Plotting variables")
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

    ###### Linear Regressor ######
    fitted_equation = build_linear_regressor(x_train,
                                             y_train)

    y_prediction_FE       = fitted_equation.predict(x_test)
    y_prediction_train_FE = fitted_equation.predict(x_train)

    # Check Coeficients
    # coefs_m = fitted_equation.coef_
    # intercept_m = fitted_equation.intercept_
    # print(intercept_m,coefs_m)

    # Check Metrics
    # Predicting the accuracy score
    # Fitted Equation:
    for index, key in enumerate(y_train.keys()):    
        FE_score = r2_score(y_test[key],y_prediction_FE[:,index])
        print("Fitted Equation:")
        print('r2 score for the fitted equation of %s is: %f'% (key,FE_score))
        print('mean_sqrd_error is: ' , mean_squared_error(y_test[key],y_prediction_FE[:,index]))
        print('root_mean_squared error is: ',np.sqrt(mean_squared_error(y_test[key],y_prediction_FE[:,index])))
        print("")

    for index, key in enumerate(y_train.keys()):    

        fitted_tendency = build_linear_regressor(
                                                 y_train[key].values.reshape(-1,1),
                                                 y_prediction_train_FE[:,index]
                                                 )

        #Check coeficients
        # coefs_b = fitted_tendency.coef_
        # intercept_b = fitted_tendency.intercept_
        # print(intercept_b,coefs_b)

        y_prediction_FT       = fitted_tendency.predict(y_test[key].values.reshape(-1,1))
        y_prediction_train_FT = fitted_tendency.predict(y_train[key].values.reshape(-1,1))

        # Check Metrics
        # Predicting the accuracy score
        # Fitted Tendency:
        FT_score = r2_score(y_test[key].values.reshape(-1,1),y_prediction_FT)
        print("Fitted Tendency Line of %s:" %key)
        print('r2 score for the fitted tendency line of %s is: %f'% (key,FT_score))
        print('mean_sqrd_error is: ' , mean_squared_error(y_test[key].values.reshape(-1,1),y_prediction_FT))
        print('root_mean_squared error is: ',np.sqrt(mean_squared_error(y_test[key].values.reshape(-1,1),y_prediction_FT)))
        print("")
    
        # # Plot Fitted Equation
        fit_path = os.path.join(cwd,"fitted_%s" % dataset)
        
        plot_fitted_equation(index,key,
                             y_train[key].values.reshape(-1,1),
                             y_test[key].values.reshape(-1,1),
                             y_prediction_train_FE[:,index],
                             y_prediction_FE[:,index],
        #                      fitted_equation,
                             fitted_tendency,
                             save_path = fit_path, 
                             save_flag = True)

        # # Save data
        ###### End of Linear Regressor ######


        