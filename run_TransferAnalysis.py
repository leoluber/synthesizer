""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber >


import numpy as np
import warnings
warnings.filterwarnings("ignore")

# custom
from Datastructure import Datastructure
from GaussianProcess import *





"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the result
"""

MOLECULES = ["Pentanone", "Butanone", "3-Pentanone", "Propanol"]
lengths = [1, 2, 4, 5, 8]
areas = []

for TRANSFER_MOLECULE in MOLECULES:
    datastructure = Datastructure(
                                synthesis_file_path = "Dataset_Transfer.csv",
                                target              = "PEAK_POS",                                  
                                wavelength_unit     = "NM",
                                monodispersity_only = False,
                                P_only              = True,
                                molecule            = TRANSFER_MOLECULE,
                                add_baseline        = True,
                                )
                                

    #%%

    # adjust the selection of training parameters
    datastructure.synthesis_training_selection  = ["AS_Pb_ratio", "Cs_Pb_ratio",]
    data_objects = datastructure.get_data()




    #%%


    # select input and target from Data objects
    inputs, targets, = [], []


    for data in data_objects:

        # if data["molecule_name"] != TRANSFER_MOLECULE:
        #     continue

        inputs.append(data["encoding"] + data["total_parameters"])


                # TARGETS
        targets.append(data["y"])



    # convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)




    #%%

    gp = GaussianProcess(
                        training_data = inputs,
                        parameter_selection = datastructure.total_training_parameter_selection, 
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

    #datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= TRANSFER_MOLECULE,)
    area = datastructure.plot_2D_contour_old(kernel = gp, molecule= TRANSFER_MOLECULE,)
    #datastructure.plot_2D_contour(kernel = gp,)

    areas.append(area/10000)
    print(f"Area under the curve: {area}")


# plot the areas
plt.show()

import matplotlib.pyplot as plt
plt.plot(lengths, areas, "s--", color="cornflowerblue")

plt.ylabel("area below 464nm (norm)")
plt.xlabel("carbon chain length")

# rotate x-axis labels
plt.xticks(rotation=45)

#ticks inside
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')

plt.show()
