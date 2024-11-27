""" Runs a Gaussian Process for list-format inputs from a Datastructure object and plots the result """
    # < github.com/leoluber >


import numpy as np
import warnings
warnings.filterwarnings("ignore")

# custom
from Datastructure import *
from GaussianProcess import *
from Preprocessor import *





"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the result
"""



datastructure = Datastructure(
                            synthesis_file_path = "Perovskite_NC_synthesis_NH_240418.csv", 
                            target              = "PEAK_POS",
                            #wavelength_filter  = [455, 464],                                        
                            wavelength_unit     = "NM",
                            monodispersity_only = False,
                            encoding            = "geometry", 
                            P_only              = True,
                            molecule            = "Isopropanol",
                            add_baseline        = True,
                            )
                            

#%%

# adjust the selection of training parameters
datastructure.synthesis_training_selection  = ["AS_Pb_ratio", "Cs_Pb_ratio",]
#datastructure.synthesis_training_selection  = ["AS_Pb_ratio", "Cs_Pb_ratio", "c (PbBr2)", "V (antisolvent)", "V (Cs-OA)", "V (PbBr2 prec.)", ]


# get data objects (either from file or from datastructure)
#datastructure.save_data_as_file(datastructure.get_data(), "data_objects")
#data_objects = datastructure.load_data_from_file("data_objects")
data_objects = datastructure.get_data()

#datastructure.plot_parameters(data_objects,)



# # use Preprocessor to select data points
# ds = Preprocessor(selection_method= ["IDEAL",], mode="EV")
# data_objects = ds.select_data(data_objects)                           
# datastructure.data  = data_objects
#data_objects  = ds.add_residual_targets_avg(data_objects)


# plot
# fwhm = [data["y"] for data in data_objects]
# peak_pos = [data["peak_pos"] for data in data_objects]
# plt.plot(peak_pos, fwhm, "o")
# plt.show()


# get parameter selection
parameter_selection = datastructure.total_training_parameter_selection


# plot the parameters
#datastructure.plot_data("Cs_As_ratio", "AS_Pb_ratio", molecule= "all")
#datastructure.plot_parameters(data_objects,)


#%%


# select input and target from Data objects
inputs, targets, sample_numbers = [], [], []
molecule, peak_pos = [], []


for data in data_objects:

    inputs.append(data["encoding"] + data["total_parameters"])
    #inputs.append(data["encoding"] + [data["peak_pos"]])

            # TARGETS
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



    # (1) LOO cross validation
#gp.choose_best_kernel()
# gp.leave_one_out_cross_validation(inputs, targets,)
# gp.regression_plot()



    # (2) 2D MAP
# gp.train()




    # (2) 3D MAP
gp.train()
datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= datastructure.flags["molecule"],)
# datastructure.plot_2D_contour("AS_Pb_ratio", "Cs_Pb_ratio", kernel = gp)




### -------------------- SCREENING ------------------- ###

#%%
def run_screening():
    As_Pb = 0.05
    c_As = 24.66
    V_As = 500
    c_Pb = 0.01
    V_Pb = (V_As * c_As) / (c_Pb * As_Pb * 10000)
    print(f"V_Pb: {V_Pb}")


    # screening 
    c_Cs = 0.2
    Cs_Pb = np.linspace(0, 1, 100)
    V_Cs = [Cs_Pb[i] * c_Pb * V_Pb  / c_Cs for i in range(len(Cs_Pb))]



    # encoding Methanol
    encoding = datastructure.encode("Methanol", "geometry")
    input = [encoding + [As_Pb, Cs_Pb[i]] for i in range(len(Cs_Pb))]
    input = np.array(input)

    # predict
    print(input[0])
    predictions = [gp.predict(element) for element in input]
    predictions = [predictions[i][0][0][0] for i in range(len(predictions))]

    # plot with plotly
    import plotly

    fig = plotly.graph_objs.Figure()
    fig.add_trace(plotly.graph_objs.Scatter(x=V_Cs, y=predictions, mode="lines", name="Prediction"))
    fig.show()

run_screening()

# %%
