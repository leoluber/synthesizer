""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber >


import numpy as np
import warnings
import pandas as pd
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

TRANSFER_MOLECULE = "Ethanol"

datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418_new.csv", 
                            spectral_file_path  = "spectrum/", 
                            monodispersity_only = True,
                            P_only              = False,
                            molecule            = "all",
                            add_baseline        = False,
                            encoding= "geometry",
                            )
                            
datastructure.read_synthesis_data()



#%%

# feature selection
features = ["AS_Pb_ratio", "Cs_Pb_ratio",] # "V_total", "V (Cs-OA)", "V (PbBr2 prec.)", "V (antisolvent)", "c (PbBr2)",]

# get training data
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="PLQY", encoding=True)


#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP",  
                    )




    # (1) LOO cross validation
# baseline = selection_dataframe["baseline"].to_numpy().astype(bool)
# #include = np.array([True if molecule == TRANSFER_MOLECULE else False for molecule in selection_dataframe["molecule_name"]]
# print(selection_dataframe["peak_pos"])
include = np.array([True if peak_pos > 455 and peak_pos < 466 else False for peak_pos in selection_dataframe["peak_pos"]])
gp.leave_one_out_cross_validation(inputs, targets, include_sample=include)
gp.regression_plot()

exit()


    # (2) 2D MAP
# gp.train()


    # (2) 3D MAP
gp.train()
gp.print_parameters()

plotter = Plotter(datastructure.processed_file_path, encoding= datastructure.encoding, selection_dataframe= selection_dataframe)

plotter.plot_correlation()
#plotter.plot_ternary(selection_dataframe= selection_dataframe, molecule= MOLECULE)
# for MOLECULE in ["Methanol", "Ethanol", "Butanol", "Cyclopentanone", "Isopropanol"]:
#     plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)

    #molecule_df = selection_dataframe[selection_dataframe["molecule_name"] == MOLECULE]
    #molecule_df = molecule_df[[ "AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", "Sample No.", "molecule_name"]]
    # write to data.csv
    #molecule_df.to_csv("data.csv", mode='a', header=False, index=True)

#plotter.plot_2D_contour_old(kernel = gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
#plotter.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
plotter.plot_parameters(var1 = "peak_pos_eV", var2 = "fwhm", color_var = "peak_pos_eV")
