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
    Runs a Gaussian Process
    and plots the result
"""

datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418_new.csv", 
                            spectral_file_path  = "spectrum/", 
                            monodispersity_only = True,
                            P_only              = True,
                            S_only              = False,
                            molecule            = "all",
                            add_baseline        = True,
                            encoding= "geometry",
                            )
                            
datastructure.read_synthesis_data()



#%%

# feature selection
features = ["AS_Pb_ratio", "Cs_Pb_ratio",] # "V_total", "c (PbBr2)", "c (Cs-OA)"] # for fwhm/plqy

# get training data
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos", encoding=True)

# print(len(inputs), len(targets))

# # save dataframe to csv
# print_dataframe = selection_dataframe[["Sample No.", "molecule_name", "monodispersity", "S/P"]]
# #sort by molecule name
# print_dataframe = print_dataframe.sort_values(by="molecule_name")
# print_dataframe.to_csv("data.csv", mode='w', header=True, index=False)
# exit()

#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP",  
                    )




#     # (1) LOO cross validation
# for MOLECULE in ["Methanol", "Ethanol", "Isopropanol", "Butanol", "Cyclopentanone",]:
#     baseline = selection_dataframe["baseline"].to_numpy().astype(bool)
#     include = np.array([True if molecule == MOLECULE else False for molecule in selection_dataframe["molecule_name"]])
#     gp.leave_one_out_cross_validation(inputs, targets, include_sample=include, baseline_list=baseline)
#     gp.regression_plot(MOLECULE)

# index if peak_pos between 455 and 464
# include = np.array([True if 455 < peak_pos < 464 else False for peak_pos in selection_dataframe["peak_pos"]])
# gp.leave_one_out_cross_validation(inputs, targets, include_sample=include,)
# gp.regression_plot() #MOLECULE)

# exit()


    # (2) 2D MAP
gp.train()
plotter = Plotter(datastructure.processed_file_path, encoding= datastructure.encoding, selection_dataframe= selection_dataframe)
plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= gp, molecule= "Methanol", library="matplotlib", selection_dataframe= selection_dataframe)
exit()


    # (2) 3D MAP
gp.train()
#gp.print_parameters()

#plotter.plot_correlation()
#plotter.plot_ternary(selection_dataframe= selection_dataframe, molecule= MOLECULE)
for MOLECULE in ["Methanol", "Ethanol", "Isopropanol", "Butanol", "Cyclopentanone",]: 
    #plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe, library="matplotlib")
    plotter.plot_2D_contour_old("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)

    molecule_df = selection_dataframe[selection_dataframe["molecule_name"] == MOLECULE]
    molecule_df = molecule_df[[ "AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", "Sample No.", "molecule_name", "monodispersity", "S/P"]]
    #write to data.csv
    #molecule_df.to_csv(f"{MOLECULE}_data_S.csv", mode='a', header=False, index=True)

#plotter.plot_2D_contour_old(kernel = gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
#plotter.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, molecule= MOLECULE, selection_dataframe= selection_dataframe)
#plotter.plot_parameters(var1 = "peak_pos_eV", var2 = "fwhm", color_var = "peak_pos_eV")
