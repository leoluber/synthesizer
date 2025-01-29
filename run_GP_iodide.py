""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result
 """
    # < github.com/leoluber >




import numpy as np
import warnings
warnings.filterwarnings("ignore")

# custom
from Datastructure import Datastructure
from GaussianProcess import *
from Preprocessor import *






"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the 
    --> in this case for the Iodide system
"""





datastructure = Datastructure(
                            synthesis_file_path = "CsPbI3_NH_LB_AS_BS_combined.csv", 
                            spectral_file_path  = "spectrum_CsPbI3/",
                            target              = "PEAK_POS",
                            PLQY_criteria       = False,
                            #wavelength_filter  = [455, 464],                                        
                            wavelength_unit     = "NM",
                            monodispersity_only = False,
                            P_only              = True,
                            molecule            = "Toluene",
                            add_baseline        = False,
                            )
                            

#%%

# adjust the selection of training parameters
datastructure.synthesis_training_selection  = ["Cs_Pb_ratio", "Pb/I",]


data_objects = datastructure.get_data()

# get parameter selection
parameter_selection = datastructure.total_training_parameter_selection



#%%


# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []
molecule, peak_pos = [], []


for data in data_objects:

    inputs.append(data["total_parameters"])
    peak_pos.append(data["peak_pos"])


            # TARGETS
    targets.append(data["y"])

plt.show()

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
# gp.leave_one_out_cross_validation(inputs, targets,)
# gp.regression_plot()



    # (2) 2D MAP
# gp.train()


    # (2) 3D MAP
gp.train()
gp.print_parameters()

datastructure.plot_data(datastructure.synthesis_training_selection[0], datastructure.synthesis_training_selection[1], kernel= gp, model = "GP", molecule= "Toluene",)
#datastructure.plot_2D_contour_old(kernel = gp, molecule= TRANSFER_MOLECULE,)
#datastructure.plot_2D_contour(kernel = gp,)

exit()

