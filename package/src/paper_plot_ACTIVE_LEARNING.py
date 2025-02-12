
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *


# SETTINGS
#NEW_MOLECULE = "Cyclopentanone"



datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418_new.csv", 
                              spectral_file_path  = "spectrum/",        
                              monodispersity_only= True,
                              P_only=True, 
                              molecule="all",
                              add_baseline= True,
                              )


datastructure.read_synthesis_data()



#%%

# feature selection
features = ["AS_Pb_ratio", "Cs_Pb_ratio", ]

# get training data
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos", encoding=True)


#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP",  
                    )


"""
_____________________________________________________________________________________

    TRAINING PROGRESS (SI)
_____________________________________________________________________________________

...

# """
for molecule in ["Methanol", "Ethanol", "Butanol", "Cyclopentanone"]:
    gp.active_learning_simulation(selection_dataframe, measured_molecule = molecule, resolution=5)

plt.show()

#datastructure.plot_data(var1="Cs_Pb_tratio", var2= "AS_Pb_ration", kernel =  gp, molecule= NEW_MOLECULE, library= "plotly",)