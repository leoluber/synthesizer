""" 
    Module:         run_GP.py
    Project:        Synthesizer: Chemistry-Aware Machine Learning for 
                    Precision Control of Nanocrystal Growth 
                    (Henke et al., Advanced Materials 2025)
    Description:    This script uses the GaussianProcess.py module to run a GP regression
                    on the Perovskite NC synthesis data and plot the results in 3D and 2D 
                    contour plots
    Author:         << github.com/leoluber >> 
    License:        MIT
    Year:           2025
"""


# -------
import numpy as np
import warnings
import os
import sys
# -------


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# custom
from package.src.GaussianProcess import GaussianProcess
from package.src.Datastructure import Datastructure
from package.plotting.Plotter import Plotter




"""
    Utilizes the GaussianProcess class to run a GP regression on the Perovskite NC synthesis data
    and plot the results in 3D and 2D contour plots
"""

# NOTE: update paths and selections here if necessary
datastructure = Datastructure(
                            synthesis_file_path = "dataset_synthesizer.csv", 
                            monodispersity_only = True,
                            P_only              = False,
                            S_only              = False,
                            molecule            = "all",
                            add_baseline        = True,
                            encoding= "geometry",
                            )


### -------- TODO: RUN THE FOLLOWING LINES ONCE ----------- ###
# reads data from the synthesis_file_path and normalizes it
# to create the processed file at data/processed/processed_data.csv
datastructure.read_synthesis_data()
### ------------------------------------------------------- ###


### -------- TODO: Specify your input features here ------- ###
features = ["AS_Pb_ratio", "Cs_Pb_ratio",] 
### ------------------------------------------------------- ###


# get training data
inputs, targets_, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos", encoding=True)
targets = np.array([target for target in targets_])


gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 

                    # select a kernel type ("RBF", "EXP", "LIN")
                    kernel_type = "EXP",  
                    )
gp.train()


# select a molecule to visualize
TEST_MOLECULE = "Methanol"
plotter = Plotter(datastructure.processed_file_path, selection_dataframe= selection_dataframe)


### ------- TODO: specify your plotted parameters here (x,y,z) ------- ###
plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= gp, molecule= TEST_MOLECULE, selection_dataframe= selection_dataframe)
plotter.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel = gp, molecule= TEST_MOLECULE, selection_dataframe= selection_dataframe)
### ------------------------------------------------------------------ ###
