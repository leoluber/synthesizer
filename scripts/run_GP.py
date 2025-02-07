""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber >


import numpy as np
import warnings
import os
import sys
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# custom
from package.src.GaussianProcess import GaussianProcess
from package.src.Datastructure import Datastructure
from package.plotting.Plotter import Plotter





"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the result
"""

MOLECULE = "Butanol"

datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418_new.csv", 
                            spectral_file_path  = "spectrum/", 
                            monodispersity_only = False,
                            P_only              = True,
                            molecule            = "all",
                            add_baseline        = False,
                            )
                            
datastructure.read_synthesis_data()



#%%

# feature selection
features = ["AS_Pb_ratio", "Cs_Pb_ratio", ]

# get training data
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos", encoding=True)


#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP", 
                    model_type  = "GPRegression",   
                    )




    # (1) LOO cross validation
#gp.choose_best_kernel()
# gp.leave_one_out_cross_validation(inputs, targets,)
# gp.regression_plot()



    # (2) 2D MAP
# gp.train()


    # (2) 3D MAP
gp.train()
gp.print_parameters()

plotter = Plotter(datastructure.processed_file_path)

#plotter.plot_correlation()
plotter.plot_ternary(selection_dataframe= selection_dataframe, molecule= MOLECULE)
#plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
#plotter.plot_2D_contour_old(kernel = gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
#plotter.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
