""" Runs a Kernel Ridge Regression for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber >


import warnings
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from KRR import Ridge
from helpers import *



#%%
"""
    Kernel Ridge Regression on a Datastructure object
    - Datastructure: object containing the data (datastructure.py)
    - Preprocessor: object for selecting data points (preprocessor.py)
    - Ridge: Kernel Ridge Regression object (KRR.py)
    - helpers: helper functions (helpers.py)

"""



datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                              
                              target = "FWHM",
                              #wavelength_filter= [420, 440],                
                              PLQY_criteria = False,
                              wavelength_unit= "NM",
                              monodispersity_only = True,
                              encoding= "one_hot",
                              P_only=False, 
                              molecule="all",
                              add_baseline= False,
                              )


#%%

# datastructure.synthesis_training_selection  = ["AS_Pb_ratio", "V (Cs-OA)",  
#                                                "c (PbBr2)",  "V (PbBr2 prec.)","V (antisolvent)", ] 
datastructure.synthesis_training_selection    = ["AS_Pb_ratio", "Cs_Pb_ratio", "c (PbBr2)"]

#datastructure.synthesis_training_selection  = []
data_objects = datastructure.get_data()

#datastructure.write_to_file(data_objects, "PLQY.txt")



""" --- use Preprocessor to select data points --- """

# ds = Preprocessor(selection_method= ["PEAK_SHAPE",], 
#                   fwhm_margin= 0, 
#                   peak_error_threshold=0.0002, 
#                   mode=datastructure.wavelength_unit)
#data_objects = ds.select_data(data_objects)   

#data_objects = ds.add_residual_targets(data_objects) 
#data_objects  = ds.add_residual_targets_avg(data_objects) 

""" ---------------------------------------------- """



#datastructure.plot_correlation()
datastructure.plot_benchmark(data_objects, highest_lowest= "lowest", color= "blue")
datastructure.plot_benchmark(data_objects, max_sample= 100, highest_lowest= "lowest", color= "red")
plt.show()
datastructure.plot_parameters( data_objects,)


#%%

# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []
peak_pos = []

for data in data_objects:

        # INPUTS
    input =  data["total_parameters"]
    inputs.append(input)

        # TARGETS
    targets.append(data["y"])
    #targets.append(data["y_res"])




### Some extra plotting

# x = [input[-1] for input in inputs]
# y = [input[-2] for input in inputs]
# z = [target for target in targets]


#3d plot with plotly
# fig = plotly.graph_objs.Figure(data=[plotly.graph_objs
#                                      .Scatter3d(x=x, y=y, z=z, 
#                                                 mode='markers', marker=dict(size=8))])
# plotly.offline.plot(fig, filename='3d-scatter.html')

#2d plot with matplotlib
# plt.scatter(x, z, c=z)
# plt.colorbar()
# plt.show()



### --------------------- KERNEL RIDGE REGRESSION ---------------------- ###

rk = Ridge(inputs, 
           targets, 
           datastructure.synthesis_training_selection, 
           kernel_type= "laplacian", 
           alpha=0.1, gamma=0.1)


# optimize hyperparameters
print("finding hyperparameters ... ")
rk.optimize_hyperparameters() 


rk.fit()
#datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", "target", kernel=rk, molecule = datastructure.flags["molecule"])
datastructure.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel=rk,)


#loo regression
rk.loo_validation(inputs, targets, datastructure.flags["molecule"])

plt.show()



#%%