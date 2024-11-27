import warnings
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *
import pandas




datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              
                              target = "FWHM",         
                              PLQY_criteria = False,
                              wavelength_unit= "EV",
                              #wavelength_filter= [465, 478],
                              monodispersity_only = True,
                              encoding= "geometry",
                              P_only= True, 
                              molecule="all",
                              add_baseline= False,
                              )



#%%

# adjust the selection of training parameters
datastructure.synthesis_training_selection    = ["AS_Pb_ratio", "Cs_Pb_ratio", ]
data_objects = datastructure.get_data()
parameter_selection = datastructure.total_training_parameter_selection



# write the PLQY data to a csv file
# plqy = [data["y"] for data in data_objects]
# peaks = [data["peak_pos"] for data in data_objects]
# df = pandas.DataFrame({"PLQY": plqy, "Peak": peaks})
# df.to_csv("PLQY_data.csv")

# exit()

#%%

"""
_____________________________________________________________________________________

    Ternary Plot (Figure ?)
_____________________________________________________________________________________

...

"""

# datastructure.plot_ternary(data_objects)
# exit()


#%%
"""
_____________________________________________________________________________________

    FWHM & PLQY (Figure 3 & 4)
_____________________________________________________________________________________

...

"""

#PLQY
datastructure.plot_parameters(data_objects,)
plt.show()

exit()




"""
_____________________________________________________________________________________

    SCREENING
_____________________________________________________________________________________

...

"""

# train the GP
# inputs  = [data["encoding"] + data["total_parameters"] for data in data_objects]
# targets = [data["y"] for data in data_objects]  
# gp = GaussianProcess(
#                     training_data = np.array(inputs),
#                     parameter_selection = parameter_selection, 
#                     targets = np.array(targets), 
#                     kernel_type = "EXP", 
#                     model_type  = "GPRegression",   
#                     )
# gp.train()


# # PLQY
# data_objects = [item for item in data_objects if item["Cs_Pb_ratio"] == 0.2  and item["molecule_name"] == "Methanol"]
# datastructure.plot_screening(data_objects=data_objects, model=gp)
# plt.show()

# exit()



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

# molecules = ["Methanol", "Ethanol", "Butanol", "Toluene", "Cyclopentanone",]

# matrix = [[[] for _ in range(len(datastructure.ml_dictionary))] for _ in range(len(molecules))]

# for i, data in enumerate(data_objects):
#     molecule = data["molecule_name"]
#     if molecule not in molecules:
#         continue
#     ml = get_ml_from_peak_pos(data["peak_pos"])
    
#     if ml is not None:
#         matrix[molecules.index(molecule)][int(ml-2)].append(data["y"])


# # delete entries with less than 5 data points
# for i in range(len(matrix)):
#     for j in range(len(matrix[i])):
#         if len(matrix[i][j]) < 1:
#             matrix[i][j] = 0
#         else:
#             #matrix[i][j] = np.mean(matrix[i][j]) # *1000
#             matrix[i][j]  = np.min(matrix[i][j])
#             #matrix[i][j] = np.max(matrix[i][j])

# # plot the matrix as heatmap
# fig, ax = plt.subplots()
# cmap = plt.get_cmap('Blues')
# cmap.set_under('grey')
# cax = ax.matshow(matrix, cmap= cmap,  vmin=0.065, vmax=0.12) #vmin = 0.01, vmax = 1, )#


# fig.colorbar(cax,) #  label="AVG PLQY",)
# plt.tick_params(axis='x', which='both', bottom=False)
# fig.set_size_inches(10, 4)


# ax.set_xticklabels([""] + list(ml_dictionary.keys()), rotation=45)
# plt.xticks(range(len(ml_dictionary)), ml_dictionary.keys())
# ax.set_yticklabels([""] + molecules)
# plt.yticks(range(len(molecules)), molecules)


# plt.tight_layout()

# plt.show()




# write to csv with pandas
#df = pandas.DataFrame(matrix, columns= ml_dictionary.keys(), index= molecules)
# df.to_csv("min_FWHM_matrix.csv")

# print(df)


# write PLQY and peak positions to csv
# plqy = [data["y"] for data in data_objects]
# peaks = [data["peak_pos"] for data in data_objects]
# df = pandas.DataFrame({"PLQY": plqy, "Peak": peaks})
# df.to_csv("PLQY_data.csv")