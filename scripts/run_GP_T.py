""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result
 """
    # < github.com/leoluber >

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
import sys
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# custom
from package.src.Datastructure import Datastructure
from package.src.GaussianProcess import GaussianProcess
from package.plotting.Plotter import Plotter
import pandas
import numpy as np



"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the 
    --> in this case for the Iodide system
"""





datastructure = Datastructure(
                            synthesis_file_path = "Synthesis_overview_PL.csv", 
                            spectral_file_path  = "spectrum_T/",     
                            monodispersity_only = True,
                            P_only              = False,
                            S_only              = False,
                            add_baseline        = False,
                            encoding= "geometry",
                            )

datastructure.read_synthesis_data()

#%%

# feature selection
features = ["Cs_Pb_ratio", "T" ,] 

# get training data
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos")


#exit()
#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP", 
                    )



    # (1) LOO cross validation
# gp.leave_one_out_cross_validation(inputs, targets,)
# gp.regression_plot()

# exit()

    # (2) 2D MAP
# gp.train()


    # (2) 3D MAP
gp.train()
gp.print_parameters()



plotter = Plotter(datastructure.processed_file_path)
#plotter.plot_correlation(selection_dataframe= selection_dataframe)
plotter.plot_data(features[0], features[1], "peak_pos", kernel= None, molecule= "Toluene", selection_dataframe=selection_dataframe)
#plotter.plot_2D_contour_old(kernel = gp, molecule= "Toluene",)
#plotter.plot_2D_contour(kernel = gp,)


