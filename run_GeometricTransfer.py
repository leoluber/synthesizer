import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# cusom
import GeometricTransfer
import Datastructure


"""
    This script is used to run the GeometricTransfer model for predicting the properties
    of perovskite NPLs for unknown antisolvents
    The model is trained on a set of synthesis data
        - >  (explained in the Datastructure.py file)
"""



### --------------------------------- INIT ---------------------------------- ###


geo_transfer = GeometricTransfer.GeometricTransfer(
                                                   molecule_training_selection= [ "Ethanol", "Butanol",],
                                                   expert=                      "SIGMOID_FILTERS",                               # "KRR", "SIGMOID_FILTERS", "GP"
                                                   predictor=                   "KRR",                                           # "MLP", "KRR", "GP"
                                                   mlp_layers=                   50,                                             # only relevant for predictor = "MLP"
                                                   data_path =                  "Perovskite_NC_synthesis_NH_240418.csv",
                                                   )

geo_transfer.viz_experts()



### ---------------------------- PREDICT NEW MOLECULE -------------------------- ###


molecule =  "Propanol"
molecule_data = Datastructure.Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv",
                                            target="PEAK_POS",
                                            output_format="LIST",
                                            wavelength_unit="NM",             
                                            monodispersity_only=True,   
                                            molecule=molecule,   
                                            )


# get the molecule data
molecule_data = molecule_data.get_data()

inputs =        [data["total_parameters"][0] for data in molecule_data]
targets =       [data["y"] for data in molecule_data]
predictions =   []


### ------------------------------ GEOMETRY TRANSFER ---------------------------- ###

#plot the model predictions
for input in inputs:
    pred = geo_transfer.forward_pass([input], molecule= molecule)[0]
    predictions.append(pred)


x_vec = np.linspace(0, 10, 300)	
y_vec = [geo_transfer.forward_pass([x], molecule = molecule)[0] for x in x_vec]


fig =   go.Figure()

fig.add_trace(go.Scatter(x= x_vec, y= y_vec, mode='lines', name='KRR'))
fig.add_trace(go.Scatter(x= inputs, y= targets, mode='markers', name='Data'))

fig.show()



