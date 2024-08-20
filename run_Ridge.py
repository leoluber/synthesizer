#%%
import warnings
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from KRR import Ridge
from helpers import *


"""
    Kernel Ridge Regression on a Datastructure object
"""

### --------------------- GET DATA ---------------------- ###

datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              
                              target = "PEAK_POS",                          # "PLQY", "FWHM", "PEAK_POS", "NPL_TYPE", "NR"
                              output_format= "LIST",
                              #wavelength_filter= [470, 495],               # [457, 466] --->   ALWAYS IN NM !!!!!
                              exclude_no_star= True,                        # exclude '*' in dataset
                              wavelength_unit= "NM",                        # "NM", "EV"
                              normalization = True,                         # normalize data, default is "True"
                              monodispersity_only = False,
                              encoding= "one_hot",                          # "one_hot", "geometry"
                              P_only= False,                                # only P in the dataset
                              )

#datastructure.synthesis_training_selection = ["AS_Pb_ratio", "V (antisolvent)",  "V (PbBr2 prec.)", "V (Cs-OA)" ]
datastructure.synthesis_training_selection = ["AS_Pb_ratio",]
data_objects = datastructure.get_data()
parameter_selection = datastructure.total_training_parameter_selection       # get parameter selection



# datastructure.plot_As_Pb_peak_pos("all", [0, 1000])
# plt.show()

# exit()
#%%

# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []

for data in data_objects:

        # INPUTS
    #input = (data["encoding"] + data["total_parameters"] )
    input = data["total_parameters"]
    inputs.append(input)

        # TARGETS
    targets.append(data["y"])

        # SAMPLE NUMBERS
    sample_numbers.append(data["sample_number"])


#%%

### --------------------- KERNEL RIDGE REGRESSION ---------------------- ###

rk = Ridge(inputs, targets, parameter_selection, kernel_type= "polynomial",alpha=0.01, gamma=0.01)  # kernel ridge regression

# optimize hyperparameters
print("finding hyperparameters ... ")
rk.optimize_hyperparameters()                                              # optimize hyperparameters



# (1) loo regression
rk.loo_validation(inputs, targets)                                        # fit model



# (2) vizualize the model
# plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')
# rk.fit()                                                                   # fit model
# rk.visualize_kernel(ax)                                                    # visualize the kernel



# (3) MAP
# sample = random.sample(inputs, 1)[0]                                      # sample data
# sample_target = targets[inputs.index(sample)]                             # get target
# sample_number = sample_numbers[inputs.index(sample)]                      # get sample number
# rk.fit()                                                                  # fit model
# rk.map_3D(sample, sample_target, sample_number,
#           "special_ratio", "polarity")



# (4) plot kernel for a single parameter
rk.fit()                                                                   # fit model
rk.plot_1D([0], 0)                                                               # plot kernel for a single parameter


plt.show()



#%%