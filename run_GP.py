""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber >


import numpy as np

# custom
from Datastructure import *
from GaussianProcess import *
from Preprocessor import *





"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the result
"""



datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418.csv", 
                            target              = "PEAK_POS",
                            #wavelength_filter  = [450, 480],                                        
                            wavelength_unit     = "NM",
                            monodispersity_only = True,
                            encoding            = "one_hot", 
                            P_only              = False,
                            molecule            = "Methanol",
                            add_baseline        = True,
                            )

#%%

# adjust the selection of training parameters
# datastructure.synthesis_training_selection = ["c (PbBr2)", "V (antisolvent)", "V (Cs-OA)", 
#                                               "V (PbBr2 prec.)", "c (Cs-OA)", "c (OlAm)", "c (OA)"]
datastructure.synthesis_training_selection   = ["AS_Pb_ratio", "Cs_Pb_ratio",]
data_objects = datastructure.get_data()


# use Preprocessor to select data points
ds = Preprocessor(selection_method= ["PEAK_SHAPE",], fwhm_margin=-0.01, peak_error_threshold=0.0002)
data_objects = ds.select_data(data_objects)                           
datastructure.data  = data_objects

# get parameter selection
parameter_selection = datastructure.total_training_parameter_selection




#%%


# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []
molecule, peak_pos = [], []


for data in data_objects:
    
    inputs.append(data["encoding"] + data["total_parameters"])

            # TARGETS
    targets.append(data["y"])



# convert to numpy arrays
inputs = np.array(inputs)
targets = np.array(targets)




#%%

gp = GaussianProcess(
                    training_data = inputs,
                    parameter_selection = parameter_selection, 
                    targets = targets, 
                    kernel_type = "EXP", 
                    model_type  = "GPRegression",   
                    )



    # (1) LOO cross validation
#gp.choose_best_kernel()
gp.leave_one_out_cross_validation(inputs, targets)
gp.regression_plot()


    # (2) 3D MAP
gp.train()
datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= datastructure.flags["molecule"])
