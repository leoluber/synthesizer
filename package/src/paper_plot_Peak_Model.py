#%%
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from GaussianProcess import GaussianProcess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from plotting.Plotter import Plotter
from helpers import *


# transfer molecule
TRANSFER_MOLECULE = "Methanol"


datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418_new.csv", 
                              spectral_file_path  = "spectrum/",        
                              monodispersity_only= True,
                              P_only=True, 
                              molecule="all",
                              add_baseline= True,
                              encoding= "combined",
                              )


datastructure.read_synthesis_data()



#%%

# feature selection
features = ["AS_Pb_ratio", "Cs_Pb_ratio", ]

# get training data and the corresponding selection dataframe
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos", encoding=True)

print(selection_dataframe[features])

#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP",  
                    )


#leave one out cross validation
#baseline and include are necessary to exclude the transfer molecule from the training set
#and to exclude the baseline from the test set
baseline = selection_dataframe["baseline"].to_numpy().astype(bool)
include = np.array([True if molecule == TRANSFER_MOLECULE else False for molecule in selection_dataframe["molecule_name"]] )



# gp.leave_one_out_cross_validation(inputs, targets, baseline, include)
# gp.regression_plot(TRANSFER_MOLECULE)

# exit()

"""
_____________________________________________________________________________________

    REGRESSION PLOT (Figure 2)
_____________________________________________________________________________________

...

# """


# molecule loo with histogram
antisolvents  = ["Methanol", "Ethanol", "Isopropanol", "Butanol", "Cyclopentanone",]
#antisolvents  = ["Butanol",]
errors = []
errors_10 = []
errors_max = []

# distribution = datastructure.get_molecule_distribution()
# min = min(distribution["Methanol"], distribution["Ethanol"], distribution["Butanol"], distribution["Cyclopentanone"])
# print(min)


for transfer_molecule in antisolvents:

    # (I):  Full Transfer; no known data on transfer_molecule 
    x, y, x_test, y_test = datastructure.get_transfer_training_data(training_selection=features, 
                                                                  selection_dataframe = selection_dataframe, 
                                                                  molecule_to_exclude = transfer_molecule,
                                                                  num_samples = 0,
                                                                  target = "peak_pos",
                                                                  encoding = True,)
    
    error= gp.validate_transfer(x, y, x_test, y_test)
    errors.append(float(error))



    # (II):  Transfer with 10 known data points on transfer_molecule
    x, y, x_test, y_test = datastructure.get_transfer_training_data(training_selection=features, 
                                                                  selection_dataframe = selection_dataframe, 
                                                                  molecule_to_exclude = transfer_molecule,
                                                                  num_samples = 10,
                                                                  target = "peak_pos",
                                                                  encoding = True,)
                                                                  
    error_10 = gp.validate_transfer(x, y, x_test, y_test)
    errors_10.append(error_10)

    extra_gp = GaussianProcess(x, y, kernel_type = "EXP")
    extra_gp.train()
    plotter = Plotter(datastructure.processed_file_path, encoding= datastructure.encoding)
    plotter.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", kernel= extra_gp, molecule= transfer_molecule, selection_dataframe= selection_dataframe)
  
    




    # (III):  Transfer with all known data points on transfer_molecule (LOO)
    baseline = selection_dataframe["baseline"].to_numpy().astype(bool)
    include = np.array([True if molecule == transfer_molecule else False for molecule in selection_dataframe["molecule_name"]] )
    err_max = gp.leave_one_out_cross_validation(inputs, targets, baseline, include)
    errors_max.append(err_max)




#plot the results by showing both the errors and the improved errors in a bar plot next to each other
fig, ax = plt.subplots()
barWidth = 0.25
r1 = np.arange(len(errors))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, errors, color='cornflowerblue', width=barWidth, edgecolor='grey', label='Errors')
plt.bar(r2, errors_10, color='coral', width=barWidth, edgecolor='grey', label='Improved Errors')
plt.bar(r3, errors_max, color='lightgreen', width=barWidth, edgecolor='grey', label='Min Errors')
plt.xlabel('Molecule', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(errors))], antisolvents)
plt.ylabel('Error', fontweight='bold')
          

for _, spine in ax.spines.items():
    spine.set_visible(False)
ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 20, top=False, bottom=False, )
ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 20, right=False, left=True, )
plt.xticks(rotation=55,)
plt.show()



# gp.regression_plot()



#%%

"""
_____________________________________________________________________________________

    3D MAP (Figure 2)
_____________________________________________________________________________________

...

"""

#datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= TRANSFER_MOLECULE, library= "plotly")
# datastructure.plot_2D_contour_old(kernel= gp, molecule= TRANSFER_MOLECULE,)
# datastructure.plot_2D_contour(kernel= gp,)

# write data to csv
# Cs_Pb_ratio = [data["Cs_Pb_ratio"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE]
# AS_Pb_ratio = [data["AS_Pb_ratio"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE]
# peak_pos    = [data["y"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE]

# df = pd.DataFrame({"Cs_Pb_ratio": Cs_Pb_ratio, "AS_Pb_ratio": AS_Pb_ratio, "peak_pos": peak_pos})
# df.to_csv("3D_MAP.csv")
