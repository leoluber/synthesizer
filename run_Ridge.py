#%%
import warnings
import random
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from KRR import Ridge
from helpers import *


"""
    Kernel Ridge Regression on a Datastructure object
    - uses 
"""

### --------------------- GET DATA ---------------------- ###

datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              
                              target = "PEAK_POS",                          # "PLQY", "FWHM", "PEAK_POS", "NPL_TYPE", "NR"
                              output_format= "LIST",
                              #wavelength_filter= [440, 510],               # [457, 466] --->   ALWAYS IN NM !!!!!
                              exclude_no_star= False,                       # exclude '*' in dataset
                              wavelength_unit= "NM",                        # "NM", "EV"
                              normalization = True,                         # normalize data, default is "True"
                              monodispersity_only = False,
                              encoding= "one_hot",                          # "one_hot", "geometry"
                              )

datastructure.synthesis_training_selection = ["Pb_Cs_ratio", "AS_Pb_ratio", "V (antisolvent)", "c (PbBr2)", "V (PbBr2 prec.)", "V (Cs-OA)", "c (Cs-OA)",] #"AS_Pb_ratio",  "Pb_Cs_ratio", "c (OlAm)", "c (OA)"] #


data_objects = datastructure.get_data()
parameter_selection = datastructure.total_training_parameter_selection       # get parameter selection


#%%

# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []

for data in data_objects:

        # INPUTS
    inputs.append(data["encoding"] + data["total_parameters"]) 

        # TARGETS
    targets.append(data["y"])

        # SAMPLE NUMBERS
    sample_numbers.append(data["sample_number"])


#%%

### --------------------- KERNEL RIDGE REGRESSION ---------------------- ###

rk = Ridge(inputs, targets, parameter_selection, kernel_type= "polynomial", alpha= 1e-8, gamma= 0.001)  # kernel ridge regression

# optimize hyperparameters
print("finding hyperparameters ... ")
rk.optimize_hyperparameters()                                              # optimize hyperparameters



# (1) loo regression
rk.loo_validation(inputs, targets)                                        # fit model



# (2) vizualize the model
#rk.fit()                                                                   # fit model
#rk.visualize_kernel(ax)                                                    # visualize the kernel



# (3) MAP
# sample = random.sample(inputs, 1)[0]                                      # sample data
# sample_target = targets[inputs.index(sample)]                             # get target
# sample_number = sample_numbers[inputs.index(sample)]                      # get sample number
# rk.fit()                                                                  # fit model
# rk.map_3D(sample, sample_target, sample_number,
#           "special_ratio", "polarity")



# (4) plot kernel for a single parameter
# rk.fit()                                                                   # fit model
# rk.plot_1D()                                                               # plot kernel for a single parameter

# plt.scatter([487, 431,], [0.77, 0.32,], color='red', label='Synthesizer')
# plt.show()



#%%