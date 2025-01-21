
import numpy as np
import matplotlib.pyplot as plt

# custom imports
from GaussianProcess import *
from Datastructure import *




"""
    This files contains the functions to subtract a fit on the toluene baseline
    from the data. The fit is done using GaussianProcess.py
"""



datastructure = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv",
                                target = "PEAK_POS",
                                PLQY_criteria = False,
                                wavelength_unit= "NM",
                                monodispersity_only = True,
                                encoding= "geometry",
                                P_only= True,
                                molecule="all",
                                add_baseline= True,
                                )


datastructure.synthesis_training_selection = ["AS_Pb_ratio", "Cs_Pb_ratio",]


def fit_baseline(baseline_data: tuple)-> np.array:

    # fit a Gaussian Process to the baseline data
    gp = GaussianProcess(training_data=baseline_data[0],
                         targets=baseline_data[1],
                         kernel_type="EXP",)
    
    gp.train()

    # evaluate the GP on the interval of [0,1]
    x_vec= np.linspace(0, 1, 100).reshape(-1, 1)
    y = [gp.predict(x)[0][0][0] for x in x_vec]
    y = np.array(y).reshape(-1)
    
    #plot
    plt.plot(baseline_data[0], baseline_data[1], "o")
    plt.plot(x_vec, y)
    plt.show()


    return y


def subtract_baseline(data_objects: list)-> list:

    # get the baseline data from the data objects
    molecule = data_objects[0]["molecule_name"]
    Cs_Pb_baseline = [item["Cs_Pb_ratio"] for item in data_objects if item["baseline"] == True and item["molecule_name"] == molecule]
    peak_baseline  = [item["peak_pos"] for item in data_objects if item["baseline"] == True and item["molecule_name"] == molecule]


    # fit the baseline
    y = fit_baseline((np.array(Cs_Pb_baseline).reshape(-1, 1), np.array(peak_baseline)))

    # subtract the baseline from the data
    for item in data_objects:
        item["peak_pos"] = item["peak_pos"] - y[int(item["Cs_Pb_ratio"] * 99)]

    return data_objects





# get data
data_objects = datastructure.get_data()
data_objects = subtract_baseline(data_objects)
inputs, targets = [], []
for item in data_objects:
    inputs.append(item["encoding"] + [item["AS_Pb_ratio"]] + [item["Cs_Pb_ratio"]] )
    targets.append(item["peak_pos"])

inputs = np.array(inputs)
targets = np.array(targets)




gp_pred = GaussianProcess(training_data=inputs,
                            targets=targets,
                            kernel_type="EXP",)

gp_pred.train()


# plot
datastructure.plot_data("Cs_Pb_ratio", "peak_pos", kernel=gp_pred, data_objects=data_objects, molecule="Methanol")