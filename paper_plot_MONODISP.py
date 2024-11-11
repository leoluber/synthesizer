import warnings
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *




datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              
                              target = "PEAK_POS",         
                              PLQY_criteria = False,
                              wavelength_unit= "NM",
                              monodispersity_only =True,
                              encoding= "one_hot",
                              P_only= False, 
                              molecule="Butanol",
                              add_baseline= True,
                              )


#%%

# adjust the selection of training parameters
datastructure.synthesis_training_selection    = ["AS_Pb_ratio", "Cs_Pb_ratio", ]
data_objects = datastructure.get_data()
parameter_selection = datastructure.total_training_parameter_selection



#%%

# select input and target from Data objects
inputs, targets, sample_numbers, baseline, include = [], [], [], [], []
peak_pos = []

for data in data_objects:

    if data["monodispersity"] == 0:
        continue

        # INPUTS
    input = data["encoding"] + data["total_parameters"]


     #Spectrum

    # spectrum = data["spectrum"]
    # plt.plot(spectrum[0], spectrum[1])
    # print(data["sample_number"])
    # plt.show()

    inputs.append(input)

        # TARGETS
    #targets.append(data["y"])
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
gp.train()
#gp.leave_one_out_cross_validation(inputs, targets, baseline, include)




#%%
"""
_____________________________________________________________________________________

    MONODISPERSITY (Figure 5)
_____________________________________________________________________________________

...

"""


datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= datastructure.flags["molecule"], library= "plotly")
datastructure.plot_2D_contour(kernel=gp)
