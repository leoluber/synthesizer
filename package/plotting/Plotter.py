
""" 
    Module:         Plotter.py
    Project:        Synthesizer: Chemistry-Aware Machine Learning for 
                    Precision Control of Nanocrystal Growth
                    (Henke et al., Advanced Materials 2025)
    Description:    Class for plotting the processed data and trained models
                    in the context of the Datastructure class
    Author:         << github.com/leoluber >> 
    License:        MIT
    Year:           2025
"""


# -----------------------------#
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import numpy as np

# custom imports
from src.Datastructure import Datastructure
# -----------------------------#



class Plotter(Datastructure):

    """ General purpose class for plotting the processed data
    
        --> Inherits from "Datastructure" class (see Datastructure.py)
        --> Can be used to plot both data and trained models
    """


    def __init__(self,
                 processed_file_path,
                 selection_dataframe = None,
                 ):
        
        # initialize the Datastructure class
        super().__init__(synthesis_file_path = processed_file_path,)

        # read the data directly from path if no specific dataframe is provided
        if selection_dataframe is not None:
            self.data_frame = selection_dataframe
        else:
            self.data_frame = pd.read_csv(processed_file_path, header=0, sep=";")

        # get labels dictionary for axis labels (you dont want to use raw column names)
        self.labels_dict = {
            "PLQY": "plqy",
            "fwhm": "fwhm (meV)",
            "peak_pos": "PL peak position (nm)",
            "AS_Pb_ratio": "As/Pb ratio ($10^4$)",
            "Cs_Pb_ratio": "Cs/Pb ratio",
            # ... add more labels as needed
        }


# ------------------------------------------------------------------
#                             KERNEL EVALUATION
# ------------------------------------------------------------------ 

    def evaluate_kernel(self, kernel, molecule) -> dict:

        """ Evaluates the GP kernel on a regular grid for visualization purposes

        RETURNS
        -------
        dict with Z (predicted values), err (uncertainty), X, Y (meshgrid), x_vec, y_vec (1D arrays)
        """

        # set the grid (and define bounds of viz. parameter space)
        y_vec = np.linspace(0, 1, 100)
        x_vec = np.linspace(0, 1, 100)
        X, Y  = np.meshgrid(x_vec, y_vec)
        input = np.c_[X.ravel(), Y.ravel()]
        
        # add encoding to the input
        encoding = self.encode(molecule, enc= self.encoding)
        input_ = [np.append(encoding, row ) for row in input]

        # evaluate the kernel on the grid
        input_ = np.array(input_)
        predict = kernel.model.predict(input_)
        Z = predict[0].reshape(X.shape)
        err = predict[1].reshape(X.shape)

        return {"Z": Z, "err": err, "X": X, "Y": Y, "x_vec": x_vec, "y_vec": y_vec}



# ------------------------------------------------------------------
#                                PLOTTING
# ------------------------------------------------------------------ 


    def plot_data(self,
                    var1=None, var2 = None, var3 = None,
                    kernel = None,
                    molecule = "all",
                    selection_dataframe = None) -> None:


        """ Simple 3D Scatter plot of the data in parameter space, for visualization;
            kernel can be plotted as well
        """

        # choose the correct data frame
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # select the data for the correct molecule
        if molecule != "all":
            df = df[df["molecule_name"] == molecule]
        
        # different treatment for the artificial data
        artificial = df[df["artificial"] == True]
        df = df[df["artificial"] == False]

        # get the data
        var_1 = df[var1]
        var_2 = df[var2]
        var_3 = df[var3]
        peak  = df["peak_pos"]
        sample_no = df["Sample No."]
        fwhm = df["fwhm"]

        # info strings to display on hover
        info = [f"No: {str(no)}  FWHM: {str(round(fwhm, 2))}" for no, fwhm in zip(sample_no, fwhm)]

        # get the artificial data
        art_1 = artificial[var1][:5]
        art_2 = artificial[var2][:5]
        art_3 = artificial[var3][:5]

        # plot data with plotly
        fig = plotly.graph_objects.Figure()
        fig.add_trace(plotly.graph_objects
                        .Scatter3d(x=var_1, y=var_2, z=var_3, mode='markers', 
                                marker=dict(size=13, opacity=0.7, color=peak,),
                                text = info,
                                ))
        
        fig.update_traces(marker=dict(cmin=400, cmax=600, colorbar=dict(title='PEAK POS'), 
                                colorscale='rainbow', color=peak, showscale=True, opacity=1),
                            textposition='top center')

        fig.add_trace(plotly.graph_objects
                        .Scatter3d(x=art_1, y=art_2, z=art_3, mode='markers', 
                                    marker=dict(size=13, opacity=1, color="black"),
                                    ))
        
        fig.update_layout(scene = dict(xaxis_title=var1+ "($10^4$)", 
                                yaxis_title=var2 ,
                                zaxis_title="Peak Position (nm)",
                                ),
                                )

        # plot model over the parameter space
        if kernel is not None:
            
            dict_ = self.evaluate_kernel(kernel, molecule)
            Z, err, X, Y, x_vec, y_vec = dict_["Z"], dict_["err"], dict_["X"], dict_["Y"], dict_["x_vec"], dict_["y_vec"]

            fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=0.8, colorscale='greys', cmin = 400, cmax = 600))

        fig.show()
        return fig


    def plot_2D_contour(self, var1, var2, target = "peak_pos",
                            kernel = None, molecule = None,
                            selection_dataframe = None) -> None:


        """ 2D contour plot of the data in parameter space
        """

        # choose the data frame
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # select the data
        df = df[df["molecule_name"] == molecule]
        df = df[df["baseline"] == False]

        # get the data 
        x = df[var1]
        y = df[var2]
        targets = df[target]

        # a contour plot of the kernel
        fig, ax = plt.subplots(figsize=(6, 4.5))

        # evaluate the kernel on the grid
        if kernel is not None:

            # evaluate the kernel on a grid
            dict_ = self.evaluate_kernel(kernel, molecule)
            Z, err, X, Y, x_vec, y_vec = dict_["Z"], dict_["err"], dict_["X"], dict_["Y"], dict_["x_vec"], dict_["y_vec"]
            c = ax.contourf(X, Y, Z, 30, cmap='gist_rainbow_r', vmin = 400, vmax = 600, zorder = 1)

            # add black lines for the contour
            ax.contour(X, Y, Z, 30, colors='black', linewidths=0.5, zorder = 2)

            # colorbar with text size 12
            cbar = fig.colorbar(c, ax=ax, label = "PEAK POS",)
            cbar.ax.tick_params(labelsize=12,)
            cbar.set_label("peak position (nm)", fontsize = 12)


        # plot the data
        ax.scatter(x, y, c = targets,  vmin = 400, vmax = 600, s = 80,
                    cmap = "gist_rainbow_r",
                    edgecolors='black', zorder = 3)
    
        # layout
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True,)

        # set x and y range 
        ax.set_ylim(0., 1.)
        ax.set_xlim(0., 1.)

        # set labels
        if var1 in self.labels_dict.keys():
            ax.set_xlabel(self.labels_dict[var1], fontsize = 12)
        else:
            ax.set_xlabel(var1, fontsize = 12)
        if var2 in self.labels_dict.keys():
            ax.set_ylabel(self.labels_dict[var2], fontsize = 12)
        else:
            ax.set_ylabel(var2, fontsize = 12)

        ax.set_title(f"{molecule}", fontsize = 14)

        plt.show()

        return None
    

    def plot_parameters(self, var1 = None, var2 = None, color_var = None) -> None:

        """
            Plots arbitrary parameters for different visualization purposes
            -> not pretty, but useful for quick checks
        """
        
        # get the dataframe
        df = self.data_frame

        # remove the baseline data
        df = df[df["baseline"] == False]
        df = df[df["monodispersity"] == True]

        # get the data
        x = df[var1]
        y = df[var2]
        color = df[color_var]
        sample_no = df["Sample No."]    


        """ basic scatter plot """
        fig, ax = plt.subplots(figsize = (4, 4))

        # custom color map# Choose original colormap
        original_cmap = plt.get_cmap('vanimo')

        # Get the original colormap colors
        n_colors = 256
        original_colors = original_cmap(np.linspace(0, 1, n_colors))
        half = n_colors // 2
        modified_colors = np.copy(original_colors)
        modified_colors[half:] = [0, 0, 0, 1]

        cmap = plt.get_cmap('gist_rainbow')
        cmap.set_under('k')
        cbar = plt.colorbar(ax.scatter(x, y, c = color, cmap= cmap, alpha = 1, s = 70, vmin = 3.10, vmax = 2.07))
        cbar.remove()
        
        # label the points with the sample number
        # for i, txt in enumerate(sample_no):
        #     ax.annotate(txt, (np.array(x)[i]+0.001, np.array(y)[i]), fontsize = 8, color = "black")
        
        # settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,) # labeltop=True, labelbottom=False)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True, ) # labelleft=True, labelright=False)


        # plot horizontal line at 0.07
        if var2 == "fwhm":
            ax.axhline(y =70, color = "black", linestyle = "dashed", linewidth = 1, alpha = 0.3)

        # set axis labels
        if var1 in self.labels_dict.keys():
            ax.set_xlabel(self.labels_dict[var1], fontsize = 12)
        else:
            ax.set_xlabel(var1, fontsize = 12)
        if var2 in self.labels_dict.keys():
            ax.set_ylabel(self.labels_dict[var2], fontsize = 12)
        else:
            ax.set_ylabel(var2, fontsize = 12)

        plt.tight_layout()
        plt.show()

        return fig, ax
