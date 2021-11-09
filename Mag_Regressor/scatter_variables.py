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