
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *



# settings
TRANSFER_MOLECULE = "Methanol"


datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                               
                              target = "PEAK_POS",
                              #wavelength_filter= [470, 480],                
                              PLQY_criteria = False,
                              wavelength_unit= "NM",
                              monodispersity_only = True,
                              encoding= "geometry",
                              P_only= True, 
                              molecule="all",
                              add_baseline= True,
                              )


#%%
# adjust the selection of training parameters
datastructure.synthesis_training_selection    = ["AS_Pb_ratio", "Cs_Pb_ratio", ]
data_objects = datastructure.get_data()
parameter_selection = datastructure.total_training_parameter_selection


# Data Preprocessing
# use Preprocessor to select data points
# ds = Preprocessor(selection_method= ["PEAK_SHAPE"], fwhm_margin=-0.01, peak_error_threshold=0.00015)
# data_objects = ds.select_data(data_objects)                           
# datastructure.data  = data_objects
# data_objects  = ds.add_residual_targets_avg(data_objects)



#%%

# select input and target from Data objects
inputs, targets, sample_numbers, baseline, include = [], [], [], [], []
peak_pos = []

for data in data_objects:

    
        # INPUTS
    input = data["encoding"] + data["total_parameters"]
    inputs.append(input)

        # TARGETS
    targets.append(data["y"])

        # BASELINE
    baseline.append(data["baseline"])
    if data["molecule_name"] == TRANSFER_MOLECULE:
        include.append(True)
    else:
        include.append(False)




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
# gp.leave_one_out_cross_validation(inputs, targets, baseline, include)
# gp.regression_plot()

# exit()

"""
_____________________________________________________________________________________

    REGRESSION PLOT (Figure 2)
_____________________________________________________________________________________

...

# """
# molecule loo with histogram
names  = []
errors = []
improved_errors = []

for transfer_molecule in ["Methanol", "Ethanol", "Butanol", "Cyclopentanone" ]:
    error = gp.molecular_cross_validation(data_objects, transfer_molecule = transfer_molecule)
    names.append(transfer_molecule)
    errors.append(error)

for transfer_molecule in ["Methanol", "Ethanol",  "Butanol", "Cyclopentanone" ]:
    include = [True if data["molecule_name"] == transfer_molecule else False for data in data_objects]
    error = gp.leave_one_out_cross_validation(inputs, targets, baseline, include)
    improved_errors.append(error)



#plot the results by showing both the errors and the improved errors in a bar plot next to each other
fig, ax = plt.subplots()
barWidth = 0.25
r1 = np.arange(len(errors))
r2 = [x + barWidth for x in r1]
plt.bar(r1, errors, color='cornflowerblue', width=barWidth, edgecolor='grey', label='Errors')
plt.bar(r2, improved_errors, color='coral', width=barWidth, edgecolor='grey', label='Improved Errors')
plt.xlabel('Molecule', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(errors))], names)
plt.ylabel('Error', fontweight='bold')
          

for _, spine in ax.spines.items():
    spine.set_visible(False)
ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 20, top=False, bottom=False, )
ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 20, right=False, left=True, )
plt.xticks(rotation=55,)
plt.show()




#gp.regression_plot()



#%%

"""
_____________________________________________________________________________________

    3D MAP (Figure 2)
_____________________________________________________________________________________

...

"""

#datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= TRANSFER_MOLECULE, library= "plotly")
datastructure.plot_2D_contour_old(kernel= gp, molecule= TRANSFER_MOLECULE,)

# write data to csv
# Cs_Pb_ratio = [data["Cs_Pb_ratio"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE]
# AS_Pb_ratio = [data["AS_Pb_ratio"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE]
# peak_pos    = [data["y"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE]

# df = pd.DataFrame({"Cs_Pb_ratio": Cs_Pb_ratio, "AS_Pb_ratio": AS_Pb_ratio, "peak_pos": peak_pos})
# df.to_csv("3D_MAP.csv")



"""
_____________________________________________________________________________________

    ON THE BEST ANTISOLVENT
_____________________________________________________________________________________

...

"""

# histogram of the best Antisolvent with regards to FWHM, PLQY, etc

#TODO