import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *
import pandas
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from plotting.Plotter import Plotter



datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418_new.csv", 
                            spectral_file_path  = "spectrum/", 
                            monodispersity_only = True,
                            P_only              = True,
                            molecule            = "all",
                            add_baseline        = True,
                            )
                            

datastructure.read_synthesis_data()



#%%

# feature selection
features = ["AS_Pb_ratio", "Cs_Pb_ratio",]

# get training data
inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, target="peak_pos", encoding=True)



plotter = Plotter(datastructure.processed_file_path, encoding= datastructure.encoding)

"""
_____________________________________________________________________________________

    Ternary Plot (Figure ?)
_____________________________________________________________________________________

...

"""
MOLECULE = "Butanol"
plotter.plot_ternary(selection_dataframe= selection_dataframe, molecule= MOLECULE)
exit()


#%%
"""
_____________________________________________________________________________________

    FWHM & PLQY (Figure 3 & 4)
_____________________________________________________________________________________

...

"""

##PLQY
plotter.plot_parameters("peak_pos", "plqy", color_var = "peak_pos")
plt.show()

exit()






"""
_____________________________________________________________________________________

    ON THE BEST ANTISOLVENT
_____________________________________________________________________________________

PLQY / FWHM Matrix

"Tol" : 'Toluene',
                                "Ac" : "Acetone",
                                "EtOH" : "Ethanol",
                                "MeOH" : "Methanol",
                                "i-PrOH" : "Isopropanol",
                                "n-BuOH" : "Butanol",
                                "n-PrOH": "Propanol",
                                "butanone" : "Butanone",
                                "CyPen" : "Cyclopentanone",
                                "CyPol" : "Cyclopentanol",
                                "HexOH" : "Hexanol",
                                "OctOH" : "Octanol",
                                "EtAc" : "EthylAcetate",
                                "MeAc" : "MethylAcetate",	
"""

molecules = ["Methanol", "Ethanol", "Isopropanol", "Butanol", "Cyclopentanone", "Toluene",] 

matrix = [[[] for _ in range(len(datastructure.ml_dictionary))] for _ in range(len(molecules))]

# iterate over selection_dataframe
for i, row in selection_dataframe.iterrows():
    molecule = row["molecule_name"]
    if molecule not in molecules:
        continue
    ml = get_ml_from_peak_pos(row["peak_pos"])
    
    if ml is not None:
        matrix[molecules.index(molecule)][int(ml-2)].append(data["y"])


# delete entries with less than n data points
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if len(matrix[i][j]) < 1:
            matrix[i][j] = 0
        else:
            # depending on the desired output (min, max, mean)
            
            #matrix[i][j] = np.mean(matrix[i][j]) # *1000
            matrix[i][j]  = np.min(matrix[i][j])
            #matrix[i][j] = np.max(matrix[i][j])



# plot the matrix as heatmap
fig, ax = plt.subplots()
cmap = plt.get_cmap('Blues')
cmap.set_under('grey')
#cmap.set_over('red')

cax = ax.matshow(matrix, cmap= cmap, vmin=0.065, vmax=0.12) 
#cax = ax.matshow(matrix, cmap= cmap,  vmin = 0.01, vmax = 1, )



# layout stuff
fig.colorbar(cax,) #  label="AVG PLQY",)
plt.tick_params(axis='x', which='both', bottom=False)
fig.set_size_inches(10, 4)

ax.set_xticklabels([""] + list(ml_dictionary.keys()), rotation=45)
plt.xticks(range(len(ml_dictionary)), ml_dictionary.keys())
ax.set_yticklabels([""] + molecules)
plt.yticks(range(len(molecules)), molecules)


plt.tight_layout()
plt.show()

