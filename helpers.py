import numpy as np
from plotly import graph_objects as go
import pandas as pd
from scipy.optimize import curve_fit

# for plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go


"""
    general helper functions, to be added to module classes
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

ml_dictionary = {"1": (402, 407),
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
    """
        Get the average sample from a list of samples
    """

    data_dim = len(inputs[0]) 
    avg_sample = []

    for index in range(data_dim):
        avg = np.mean([data[index] for data in inputs])
        avg_sample.append(avg)

    return avg_sample


#### ------------------------------------ OTHER ------------------------------------ ###

def nm_to_ev(nm) -> float:
    return 1240/nm

def ev_to_nm(ev) -> float:
    return 1240/ev

def surface_proportion(nm) -> float:
    """
        Get the surface proportion for NPLs from a given wavelength (estimation)
    """
    prop = 0
    for key in ml_dictionary:
        if ml_dictionary[key][0] <= nm:
            prop =  2/ (float(key) + 1)
        
    return 1 - prop


def get_perfect_peak(peak_in_nm, sigma = 40) -> list:
    """
        Get a perfect peak for a given peak position and sigma
    """

    f = lambda x: np.exp((-x**2) /(2* sigma**2) )

    x = np.linspace(400, 600, 150)
    y = [round(f(i-peak_in_nm),3) for i in x]

    y = compress_spectrum(spectrum=y)
    
    return y


def compress_spectrum(spectrum, factor = 1) -> list:
    """
        Compress a spectrum by a given factor
    """

    compressed_spectrum = []

    while len(spectrum) >= factor:
        compressed_spectrum.append(np.mean(spectrum[:factor]))
        spectrum = spectrum[factor:]

    return compressed_spectrum


#### ------------------------------------ VISUALIZATION ------------------------------------ ###

def viz_regression(val_loader, model):
    """ 
        Plot regression from val_loader and model   
    """

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
    """
        Plot regression from true and predicted values (specifically for DIME-NET)
    """

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




#### -------------------------------------- PLOTTING -------------------------------------- ###

def plot_critical_As_Pb_ratio():

    """
        Plot the critical As/Pb ratio over molecule properties
        IGNORE THIS FUNCTION, JUST FOR TESTING
    """


    data = [{"molecule" : "Methanol",       "dipole": 1.70, "relative_polarity" : 0.762, "hansen": 22.3, "chain_length" : 0, "fit": 54,  "cubes": 40, "diffusivity": 2.87},
            {"molecule" : "Ethanol",        "dipole": 1.68, "relative_polarity" : 0.654, "hansen": 19.4, "chain_length" : 1, "fit": 144, "cubes": 133, "diffusivity": 1.15},
            {"molecule" : "Propanol",       "dipole": 1.65, "relative_polarity" : 0.617, "hansen": 17.4, "chain_length" : 2, "fit": 183, "cubes": 178, "diffusivity": 0.75},
            {"molecule" : "Butanol",        "dipole": 1.66, "relative_polarity" : 0.586, "hansen": 15.8, "chain_length" : 3, "fit": 249, "cubes": 257, "diffusivity": 0.56},
          #  {"molecule" : "Acetone",        "dipole": 2.86, "relative_polarity" : 0.355, "hansen": 7.0,                      "fit": 300, "cubes": 500},  # estimated, don't use for anything important
          #  {"molecule" : "Cyclopentanone", "dipole": 3.30, "relative_polarity" : 0.269, "hansen": 5.2,                      "fit": 203, "cubes": 533},
            {"molecule" : "Octanol",        "dipole": 1.68, "relative_polarity" : 0.537, "hansen": 11.2, "chain_length" : 8, "fit": 381, "diffusivity": 0.07},
          #  {"molecule" : "Hexanol",        "dipole": 1.60, "relative_polarity" : 0.559, "hansen": 12.5, "chain_length" : 6, "diffusivity": 0.18},
          # {"molecule" : "Isopropanol",    "dipole": 1.58, "relative_polarity" : 0.546, "hansen": 16.4, "chain_length" : 3, "fit": 300, "cubes": 167},  # estimated lower bound, could be higher
          # {"molecule" : "Toluene",        "dipole": 0.38, "relative_polarity" : 0.099, "hansen": 2,                        "fit": 1000, "cubes": 1000},  # estimated, don't use for anything important

            ]


    #chain_length = [data_point["chain_length"] for data_point in data]
    hansen =            [data_point["hansen"] for data_point in data]
    dipole_moment =     [data_point["dipole"] for data_point in data]
    relative_polarity = [data_point["relative_polarity"] for data_point in data]
    fit =               [data_point["fit"] for data_point in data]
    diffusivity =       [data_point["diffusivity"]*10 for data_point in data]
    #cubes =             [data_point["cubes"] for data_point in data]

    
    # fitting

    func = lambda x, a, b, c: a/ (x + b) + c
    bounds = ([1000, -50, -50], [5000, 50, 50])
    popt, pcov = curve_fit(func, fit, diffusivity, bounds= bounds)
    print(popt)
    x_vec = np.linspace(50, 400, 100)
    y_vec = func(x_vec, *popt)


    fig = go.Figure()

    fig.add_trace(go.Scatter(x = fit, y = diffusivity, mode = 'markers', name = 'diffusivity * 10', marker = dict(size = 10)))
    fig.add_trace(go.Scatter(x = fit, y = hansen, mode = 'markers', name = 'hansen', marker = dict(size = 10)))
    fig.add_trace(go.Scatter(x = x_vec, y = y_vec, mode = 'lines', name = 'Diffusion fit: a/(x + b) + c', marker = dict(size = 10)))


    fig.update_layout(title = 'Critical As/Pb ratio over molecule properties',
                      legend_title = 'Method',
                      )
    
    fig.show()
    fig.write_html("diff_hansen_critical_As_Pb_ratio.html")



if __name__ == "__main__":
    plot_critical_As_Pb_ratio()