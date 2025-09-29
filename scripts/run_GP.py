""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber/synthesizer >


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
    Utilizes the GaussianProcess class to run a GP regression on the Perovskite NPL synthesis data
    and plot the results in 3D and 2D contour plots
"""

# TODO: update paths and selections here if necessary

datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418_new.csv", 
                            spectral_file_path  = "spectrum/", 
                            monodispersity_only = True,
                            P_only              = False,
                            S_only              = False,
                            molecule            = "all",
                            add_baseline        = True,
                            encoding= "geometry",
                            )


### -------- TODO: RUN THE FOLLOWING LINES ONCE ----------- ###
# datastructure.read_synthesis_data()
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
                    kernel_type = "EXP",  
                    )
gp.train()


TEST_MOLECULE = "Methanol"
plotter = Plotter(datastructure.processed_file_path, encoding= datastructure.encoding, selection_dataframe= selection_dataframe)

### ------- TODO: specify your plotted parameters here ------- ###
plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= gp, molecule= TEST_MOLECULE, library="plotly", selection_dataframe= selection_dataframe)
plotter.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel = gp, molecule= TEST_MOLECULE, selection_dataframe= selection_dataframe)
### --------------------------------------------------------- ###
