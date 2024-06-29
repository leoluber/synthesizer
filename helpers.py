import numpy as np
from plotly import graph_objects as go
import pandas as pd

#extras for plotting
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

def get_avg_sample(inputs):
    data_dim = len(inputs[0]) 
    avg_sample = []
    for index in range(data_dim):
        avg = np.mean([data[index] for data in inputs])
        avg_sample.append(avg)
    return avg_sample


#### ------------------------------------ OTHER ------------------------------------ ###

def nm_to_ev(nm):
    return 1240/nm

def ev_to_nm(ev):
    return 1240/ev

def surface_proportion(eV):
    wavelength = ev_to_nm(eV)
    for key in ml_dictionary:
        if ml_dictionary[key][0] <= wavelength <= ml_dictionary[key][1]:
            return 2/ (float(key) + 1)
    return 0

def get_perfect_peak(peak_in_nm, sigma = 40):
    f = lambda x: np.exp((-x**2) /(2* sigma**2) )
    x = np.linspace(400, 600, 150)
    y = [round(f(i-peak_in_nm),3) for i in x]
    y = compress_spectrum(spectrum=y)
    return y


def compress_spectrum(spectrum, factor = 1):
    compressed_spectrum = []
    while len(spectrum) >= factor:
        compressed_spectrum.append(np.mean(spectrum[:factor]))
        spectrum = spectrum[factor:]
    return compressed_spectrum


#### ------------------------------------ VISUALIZATION ------------------------------------ ###

# visualize the loss over all epochs, should be replaced by library functions
# or at least upgrade to plotly   +   also implement confusion matrix
def viz_loss(train_loss, val_loss):                                         
    x = [x for x in range(len(train_loss))]
    plt.plot(x, train_loss, label = 'train loss')
    plt.plot(x, val_loss, label = 'val loss')
    plt.show()


# plot regression data from model and validation set
def viz_regression(val_loader, model):
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


# plot regression from true and predicted values
def viz_regression_scatter(true_values, pred_values):
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


# confusion matrix
def print_confusion_matrix(data_objects, trains_split, model):
    confusion_matrix = np.zeros((2,2))
    for data_object in data_objects[trains_split:]:
        if model(data_object).item() < 0.5:
            y_pred = 0
        else: y_pred = 1
        y = int(data_object.y.item())
        confusion_matrix[y_pred][y] += 1
    print(confusion_matrix)
  

    
# logic for plotting molecules from atom types and coordinates (discouraged, use graph inherent plotting functionality)
def plot_molecule(molecule):
    atoms, coordinates = molecule
    x_vec, y_vec, z_vec, colors = [],[],[],[]
    for atom in atoms:
        colors.append(atom_types[atom]*10+40)
    for x,y,z in coordinates:
        x_vec.append(x)
        y_vec.append(y)
        z_vec.append(z)
    fig = go.Figure(data=[go.Scatter3d(x = x_vec, y = y_vec, z = z_vec, mode = 'markers', 
                                       marker=dict(size = 40, color = colors, colorscale = 'Viridis', opacity = 1))])
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-5,5],),
            yaxis = dict(nticks=4, range=[-4,4],),
            zaxis = dict(nticks=4, range=[-4,4],),
            ),
    )
    fig.show()


#### -------------------------------------- PLOTTING -------------------------------------- ###

def plot_critical_As_Pb_ratio():
    """
        Plot the critical As/Pb ratio over molecule properties
    """
    data = [{"molecule" : "Methanol", "dipole": 1.7, "relative_polarity" : 0.762, "hansen": 22.3, "chain_length" : 0, "critical_470nm" : 38, "critical_480nm" : 56,},
            {"molecule" : "Ethanol",  "dipole": 1.68, "relative_polarity" : 0.654, "hansen": 19.4, "chain_length" : 1, "critical_470nm" : 131, "critical_480nm" : 139,},
            {"molecule" : "Propanol", "dipole": 1.65, "relative_polarity" : 0.617, "hansen": 17.4, "chain_length" : 2, "critical_470nm" : 177, "critical_480nm" : 186,},
            {"molecule" : "Butanol",  "dipole": 1.66, "relative_polarity" : 0.586, "hansen": 15.8, "chain_length" : 3, "critical_470nm" : 242, "critical_480nm" : 247,},
            {"molecule" : "Cyclopentanone", "dipole": 3.3, "relative_polarity" : 0.269, "hansen": 5.2, "critical_470nm" : 83, "critical_480nm" : 156,},
            {"molecule" : "Acetone",  "dipole": 2.86, "relative_polarity" : 0.355, "hansen": 7, "critical_470nm" : 210, "critical_480nm" : 300,},
            ]


    #chain_length = [data_point["chain_length"] for data_point in data]
    hansen = [data_point["hansen"] for data_point in data]
    dipole_moment = [data_point["dipole"] for data_point in data]
    #chain_length = [data_point["chain_length"] for data_point in data]
    relative_polarity = [data_point["relative_polarity"] for data_point in data]
    critical_470nm = [data_point["critical_470nm"] for data_point in data]
    #critical_480nm = [data_point["critical_480nm"] for data_point in data]

    PARAMETER = dipole_moment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = PARAMETER, y = critical_470nm, mode = 'markers', name = '470nm', marker = dict(size = 10)))
    #fig.add_trace(go.Scatter(x = PARAMETER, y = critical_480nm, mode = 'markers', name = '480nm', marker = dict(size = 10)))
    fig.update_layout(title = 'Critical As/Pb ratio over molecule properties',
                      xaxis_title = 'dipole_moment',
                      yaxis_title = 'Critical As/Pb ratio',
                      legend_title = 'Peak position',
                      )
    fig.show()



if __name__ == "__main__":
    plot_critical_As_Pb_ratio()