import warnings
import numpy as np
warnings.filterwarnings('ignore')

# custom
from Datastructure import Datastructure
from Preprocessor import Preprocessor
from GaussianProcess import GaussianProcess
from helpers import *
import pandas as pd
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



#%%

gp = GaussianProcess(
                    training_data = inputs,
                    targets = targets, 
                    kernel_type = "EXP",   
                    )
gp.train()


"""
__________________________________________________________

    MONODISPERSITY (Figure 5)
_____________________________________________________________________________________

...

"""


#datastructure.plot_data("AS_Pb_ratio", "Cs_Pb_ratio", kernel= gp, model = "GP", molecule= datastructure.flags["molecule"], library= "plotly")
#datastructure.plot_2D_contour(kernel=gp)
datastructure.plot_2D_contour_old(kernel=gp, molecule= datastructure.flags["molecule"])