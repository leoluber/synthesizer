"""File containing helper functions for the project"""
   # < gihub.com/leoluber >


import numpy as np
from plotly import graph_objects as go
from scipy.optimize import curve_fit
from typing import Literal

# for plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go



"""
    General helper functions, to be added to module classes at some point
    mostly visualization and unecessary logic :)
"""
    


# dictionaries
molecule_dictionary = {"Tol" : 'Toluene',
                       "EtAc" : "EthylAcetate",
                       "MeAc" : "MethylAcetate",
                       "Ac" : "Acetone",
                       "EtOH" : "Ethanol",
                       "MeOH" : "Methanol",
                       "i-PrOH" : "Isopropanol",
                       "n-BuOH" : "Butanol",
                       "t-BuOH" : "Tert-Butanol",
                       "n-PrOH": "Propanol",
                       "ACN" : "Acetonitrile",
                       "DMF" : "Dimethylformamide",
                       "DMSO" : "Dimethylsulfoxide",
                       "butanone" : "Butanone",
                       "CyPen" : "Cyclopentanone"}


ml_dictionary = {#"1": (402, 407),
                    "2": (430, 437),
                    "3": (457, 466),
                    "4": (472, 481),
                    "5": (484, 489),
                    "6": (491, 497),
                    "7": (498, 504),
                    "8": (505, 509),
                    "9": (510, 525),}

atom_types = {'H' : 1,
              'C' : 6,
              'O' : 8,
              'N' : 7,
              'S'  : 16
              }

atom_types_inv_map = {v: k for k, v in atom_types.items()}   #inverse dictionary     number --> atom


#### ------------------------------------ DATA ------------------------------------ ###

def get_avg_sample(inputs) -> list:
    
    """ Get the average sample from a list of samples """

    data_dim = len(inputs[0]) 
    avg_sample = []

    for index in range(data_dim):
        avg = np.mean([data[index] for data in inputs])
        avg_sample.append(avg)

    return avg_sample


#### ------------------------------------ OTHER ------------------------------------ ###

def nm_to_ev(nm) -> float:
    return 1239.840/nm


def ev_to_nm(ev) -> float:
    return 1239.840/ev


def surface_proportion(nm) -> float:

    """ Get the surface proportion for NPLs from a given wavelength (estimation) """

    prop = 0
    for key in ml_dictionary:
        if ml_dictionary[key][0] <= nm:
            prop =  2/ (float(key) + 1)
        
    return 1 - prop


def get_perfect_peak(peak_in_nm, sigma = 40) -> list:
    
    """ Get a perfect peak for a given peak position and sigma """

    f = lambda x: np.exp((-x**2) /(2* sigma**2) )

    x = np.linspace(400, 600, 150)
    y = [round(f(i-peak_in_nm),3) for i in x]

    y = compress_spectrum(spectrum=y)
    
    return y


def compress_spectrum(spectrum, factor = 1) -> list:
    
    """ Compress a spectrum by a given factor """

    compressed_spectrum = []

    while len(spectrum) >= factor:
        compressed_spectrum.append(np.mean(spectrum[:factor]))
        spectrum = spectrum[factor:]

    return compressed_spectrum


def find_lowest(data_objects) ->  list:
        
    """ Find the lowest target value for each peak_pos
    in a list of data objects """

    x = [None for _ in ml_dictionary]
    y = [float("inf") for _ in ml_dictionary]
    sample_numbers = [None for _ in ml_dictionary]

    for data in data_objects:
        for i, key in enumerate(ml_dictionary):
            if ev_to_nm(data["peak_pos"]) in range(ml_dictionary[key][0], ml_dictionary[key][1]):
                if data["y"] <= y[i]:
                    y[i] = data["y"]
                    x[i] = data["peak_pos"]
                    sample_numbers[i] = data["sample_number"]

    # clean up
    x = [i for i in x if i is not None]
    y = [i for i in y if i != float("inf")]

    print(y, sample_numbers)


    return x, y


def get_ml_from_peak_pos(peak_pos) -> int:

    """ Get the ML from a peak position """

    if peak_pos < 10:
        peak_pos = ev_to_nm(peak_pos)

    for key in ml_dictionary:
        #if peak_pos in range(ml_dictionary[key][0], ml_dictionary[key][1]):
        if peak_pos <= ml_dictionary[key][1]:
            return int(key)
        
    return None





#### ------------------------------------ VISUALIZATION ------------------------------------ ###

def viz_regression(val_loader, model):

    """ Plot regression from val_loader and model"""

    x,y = [],[]
    for data in val_loader:
        x.append(data.y.item())
        y.append(model(data).item())
        
    min_max = min(min(x), min(y)), max(max(x), max(y))
    plt.scatter(x, y,)

    plt.xlim(min_max)
    plt.ylim(min_max)
    plt.xlabel('true values (norm)')
    plt.ylabel('pred values (norm)')
    plt.title('Regression Chart')

    plt.show()



def viz_regression_scatter(true_values, pred_values):
    
    """ Plot regression from true and predicted values (specifically for DIME-NET) """

    x,y = [],[]

    for i in range(len(true_values)):
        x.append(true_values[i].item())
        y.append(pred_values[i].item())
    
    min_max = min(min(x), min(y)), max(max(x), max(y))
    plt.scatter(x, y,)

    plt.xlim(min_max)
    plt.ylim(min_max)
    plt.xlabel('true value (norm)')
    plt.ylabel('pred value (norm)')
    plt.title('DIME-NET Regression Chart')

    plt.show()




### ---------------- PHYSICS ----------------- ###

def surface_proportion(peak_pos, mode: Literal['EV', 'NM'], l = 20) -> float:	

    """ Get the surface proportion for NPLs from a given wavelength (estimation) """

    if mode == 'EV':
        nm = ev_to_nm(peak_pos)
    else:	
        nm = peak_pos	


    prop = 0
    # for key, value in ml_dictionary.items():
    #     if value[0] <= nm:
    #         surface = 2*l**2 + 40*int(key)
    #         full_layer   = l**2*4 + 2 * l
    #         half_layer   = l**2*3 + 2 * l

    #         prop = 1 -(surface / (full_layer + half_layer * (int(key)-1)))

    for key, value in ml_dictionary.items():
        if value[1] > nm:
            total = l**2 * int(key)
            surf  = total - (l-2)**2 * (int(key)-2)
            internal = (l-2)**2 * (int(key)-2)

            prop = internal / total
            return prop

        
    return prop





############### ---------------- VIZ ----------------- ################

def plot_covariance():

    """ Plot the covariance matrix of a kernel """

    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)

    kernel = lambda x, y, sigma, l: sigma**2 * np.exp(- abs(x-y) / (2*l**2))

    Z = kernel(X, Y, 1, 0.3)

    # heatmap matplotlib
    plt.imshow(Z, cmap='Greys', interpolation='nearest')
    
    # no ticks or labels
    plt.axis('off')
    plt.show()

    