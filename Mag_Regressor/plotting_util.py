import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def scatter_variables(df,x_vars,y_vars, save_path=None, save_flag = False):

    for y_var in y_vars:
      for x_var in x_vars:
        
        # Graph formatting
        plt.title("%s X %s" %(x_var, y_var))
        plt.grid(True, which='major', axis='both')
        plt.xlabel(x_var)
        plt.ylabel(y_var)

        # Graph plotting
        # plt.plot(cm_dataframe[var],cm_dataframe[predicted_variable], 'x')
        plt.scatter(df[x_var],df[y_var], s = 12, marker = "x")
        if save_flag:
            #Verify if save directory exists, if not, create it
            try:
                os.makedirs(save_path)
            except:
                print('images folder already exist')
            plt.savefig(os.path.join(save_path,"%sX%s.jpg" %(x_var, y_var)))
            plt.close()
        else:    
            plt.show()
            plt.close()
            print("")

def plot_fitted_equation(y_train,
                         y_test,
                         LR,
                         LR_2):
    

    y_prediction       = LR.predict(y_test)
    y_prediction_train = LR.predict(y_train)

    # Creating "Upper", "Lower", and "Middle" Tendency line
    # intervalo = np.linspace(np.min(y_train), np.max(y_train), 50)
    # imagem_t = np.reshape((intercept_b + coefs_b*intervalo),(50))
    # imagem_u = np.reshape((intercept_b + 0.92*coefs_b*intervalo),(50))
    # imagem_l = np.reshape((intercept_b + 1.08*coefs_b*intervalo),(50))
    # print(intervalo.shape)
    # print(imagem.shape)

    plt.figure(num = 1, figsize=(12,12))

    scatter_test  = plt.scatter(y_test,y_prediction, s = 20, c="g", marker = "x")
    scatter_train = plt.scatter(y_train,y_prediction_train, s = 12, marker = "o", facecolors = "none", edgecolors="b")

    # tendencia = plt.plot(intervalo, imagem_t, "k--")
    # tendencia_u = plt.plot(intervalo, imagem_u, c = "tab:gray", linestyle = "--")
    # tendencia_l = plt.plot(intervalo, imagem_l, c = "tab:gray", linestyle = "--")

    # Graph formatting
    plt.title("B numérico X B regressor")
    plt.grid(True, which = "major", axis = "both",linestyle="--")
    plt.xlabel("B numérico")
    plt.ylabel("B regressor")
    plt.xlim((0.8,2))
    plt.ylim((0.8,2))

    # plt.legend((scatter_test,scatter_train),("B teste","B treino"),loc= "upper right")
    plt.legend(("Linha de Tendência","B teste","B treino"),loc= "upper right")

    plt.show()
    plt.close()