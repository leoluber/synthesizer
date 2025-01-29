
import warnings
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *


# SETTINGS
#NEW_MOLECULE = "Cyclopentanone"



datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              
                              target = "PEAK_POS",               
                              PLQY_criteria = False,
                              wavelength_unit= "NM",
                              monodispersity_only= True,
                              encoding= "geometry",
                              P_only=True, 
                              molecule="all",
                              #molecule= NEW_MOLECULE,
                              add_baseline= True,
                              )




#%%
# adjust the selection of training parameters
datastructure.synthesis_training_selection    = ["AS_Pb_ratio", "Cs_Pb_ratio", ]
data_objects = datastructure.get_data()
parameter_selection = datastructure.total_training_parameter_selection




# select input and target from Data objects
inputs, targets, sample_numbers, baseline, include = [], [], [], [], []
peak_pos = []



for data in data_objects:

        # INPUTS
    input = data["encoding"] + data["total_parameters"]
    inputs.append(input)

        # TARGETS
    targets.append(data["y"])



# convert to numpy arrays
inputs = np.array(inputs)
targets = np.array(targets)


#%%

gp = GaussianProcess(training_data = inputs,
                    targets = targets,
                    kernel_type = "EXP", 
                    model_type  = "GPRegression",   
                    )
"""
_____________________________________________________________________________________

    TRAINING PROGRESS (SI)
_____________________________________________________________________________________

...

# """
# for molecule in ["Methanol", "Ethanol", "Butanol", "Cyclopentanone"]:
#     gp.active_learning_simulation(data_objects, measured_molecule = molecule, resolution=5)

# plt.show()

#datastructure.plot_data(var1="Cs_Pb_tratio", var2= "AS_Pb_ration", kernel =  gp, molecule= NEW_MOLECULE, library= "plotly",)