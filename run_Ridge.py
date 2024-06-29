#%%
from Datastructure import Datastructure
from KRR import Ridge
from helpers import *
import warnings
warnings.filterwarnings('ignore')


"""
    Kernel Ridge Regression on a Datastructure object
    - uses scikit-learn's KernelRidge
"""

### --------------------- GET DATA ---------------------- ###

datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              target = "PEAK_POS",                              # "PLQY", "FWHM", "PEAK_POS", "NPL_TYPE"
                              output_format= "LIST",
                              #wavelength_filter= [470, 500],               # [457, 466] --->   ALWAYS IN NM !!!!!
                              #exclude_no_star= True,                       # exclude '*' in dataset
                              wavelength_unit= "NM",                        # "NM", "EV"
                              normalization=False,                          # normalize data, default is "True"
                              monodispersity_only = True,
                              )

datastructure.synthesis_training_selection = ["AS_Pb_ratio",] # "V (antisolvent)", "c (PbBr2)", "V (PbBr2 prec.)", "V (Cs-OA)", ] #  "AS_Pb_ratio", "c (Cs-OA)", "Pb_Cs_ratio", "c (OlAm)", "c (OA)"] #


data_objects = datastructure.get_data()
#datastructure.normalize_target(data_objects)                                                        # normalize target (new functionality)
parameter_selection = datastructure.total_training_parameter_selection                               # get parameter selection

#%%

# plot As-Pb peak position relation
molecule_list = ["Butanol", "Propanol", "Ethanol", "Methanol", "Acetone", "Cyclopentanone",]
#molecule_list = ["Ethanol",]
for molecule in molecule_list:
    fig, ax = datastructure.plot_As_Pb_peak_pos(molecule)
    plt.show()


    # plot data distribution
#datastructure.plot_avg_target()                                                                    # plot average target distribution
#fig, ax = datastructure.plot_data("V (antisolvent)", "c (PbBr2)", "V (PbBr2 prec.)", parameter_selection)               # plot data distribution
#fig, ax = datastructure.plot_data("V (antisolvent)", "V (Cs-OA)", "V (PbBr2 prec.)", parameter_selection)               # plot data distribution
#plt.show()


# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []
for data in data_objects:

        # INPUTS
    inputs.append(data.total_parameters)
    #inputs.append([data.peak_pos])
    #inputs.append([data.fwhm])
    #inputs.append(data.one_hot_molecule + data.total_parameters)

        # TARGETS
    targets.append(data.y) 
    sample_numbers.append(data.sample_number)


### --------------------- KERNEL RIDGE REGRESSION ---------------------- ###

rk = Ridge(inputs, targets, parameter_selection, kernel_type= "polynomial", alpha= 1e-8, gamma= 0.001)  # kernel ridge regression

    # optimize hyperparameters
#print("finding hyperparameters ... ")
#rk.optimize_hyperparameters()                                              # optimize hyperparameters



    # (1) loo regression
#rk.optimize_inputs(5)                                                      # optimize the used inputs
#rk.loo_validation(inputs, targets)                                        # fit model
#rk.plot_regression()                                                      # plot regression



    # (2) vizualize the model
#rk.fit()                                                                   # fit model
#rk.visualize_kernel(ax)                                                    # visualize the kernel



    # (3) MAP
# sample = random.sample(inputs, 1)[0]                                     # sample data
# sample_target = targets[inputs.index(sample)]                            # get target
# sample_number = sample_numbers[inputs.index(sample)]                     # get sample number
# rk.fit()                                                                 # fit model
# rk.map_3D(sample, sample_target, sample_number,
#           "special_ratio", "polarity")



    # (4) plot kernel for a single parameter
#rk.fit()                                                                   # fit model
#rk.plot_1D()                                                               # plot kernel for a single parameter

# plt.scatter([487, 431,], [0.77, 0.32,], color='red', label='Synthesizer')
# #plt.legend()
#plt.show()



    # (5) predict new data
# rk.fit()

# # new data
# As_V = []
# Pb_c = []
# Pb_V = []
# Cs_V = []

# # normalize data
# As_V = [x / datastructure.max_min["V (antisolvent)"][0]     for x in As_V]
# Pb_c = [x / datastructure.max_min["c (PbBr2)"][0]           for x in Pb_c]
# Pb_V = [x / datastructure.max_min["V (PbBr2 prec.)"][0]     for x in Pb_V]
# Cs_V = [x / datastructure.max_min["V (Cs-OA)"][0]           for x in Cs_V]

# As_Pb_ratio = [x/(y*z) for x, y, z in zip(As_V, Pb_c, Pb_V)]
# As_Pb_ratio = [x / datastructure.max_min["AS_Pb_ratio"][0] for x in As_Pb_ratio]

# one_hot = [0] * len(datastructure.molecule_names)
# one_hot[datastructure.molecule_names.index("Propanol")] = 1

# for r, a, b, c, d in zip(As_Pb_ratio, As_V, Pb_c, Pb_V, Cs_V):
#     print(rk.predict([one_hot + [r, a, b, c, d]]))                      # predict new data





#%%