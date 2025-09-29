""" 
    Project:     synthesizer
    File:        Datastructure.py
    Description: Defines the Plotter class (child class of Datastructure) 
                 for visualizing the processed data
    Author:      << github.com/leoluber >> 
    License:     MIT
"""


import matplotlib.pyplot as plt
import pandas as pd
import plotly
import numpy as np
from matplotlib.colors import ListedColormap

# custom imports
from src.Datastructure import Datastructure
from src.helpers import ev_to_nm, nm_to_ev

# darkmode
plt.style.use('dark_background')



class Plotter(Datastructure):

    """ General purpose class for plotting the processed data in various ways
    
        --> Inherits from Datastructure
    """


    def __init__(self,
                 processed_file_path,
                 encoding = "combined",
                 selection_dataframe = None,
                 ):


        # read the data
        if selection_dataframe is not None:
            self.data_frame = selection_dataframe
        else:
            self.data_frame = pd.read_csv(processed_file_path, header=0, sep=";")


        self.encoding = encoding

        # paths to the data
        self.data_path_raw =            "data/raw/"
        self.encoding_path=             self.data_path_raw + "molecule_encoding.json"
        self.molecule_dictionary_path = self.data_path_raw + "molecule_dictionary.json"
        self.ml_dictionary_path =       self.data_path_raw + "ml_dictionary.json"
        self.global_attributes_path =   self.data_path_raw + "AntisolventProperties.csv"
        self.geometry_path =            self.data_path_raw + "molecule_geometry.json"

        # get encodings
        self.molecule_dictionary, self.ml_dictionary, self.encoding_dictionary = self.get_dictionaries()

        # molecule attributes
        self.global_attributes_df =  pd.read_csv(self.global_attributes_path, 
                                                 delimiter= ';', header= 0)
        
        # get labels dictionary for axis labels
        self.labels_dict = {
            "PLQY": "plqy",
            "fwhm": "fwhm (meV)",
            "peak_pos_ev": "energy (eV)",
            "peak_pos": "PL peak position (nm)",
            "sigma": "$\\sigma$ (meV)",
            "gamma1": "$\\gamma_1$ (meV)",
            "gamma2": "$\\gamma_2$ (meV)"
        }




### ---------------------------- INIT RELATED ---------------------------- ###

    def evaluate_kernel(self, kernel, molecule) -> dict:

        """ Evaluates the kernel on a grid for visualization purposes
        """

        # set the grid (and define bounds of parameter space)
        y_vec = np.linspace(0, 1, 100)
        x_vec = np.linspace(0, 1, 100)
        X, Y  = np.meshgrid(x_vec, y_vec)
        input = np.c_[X.ravel(), Y.ravel()]
        
        # add self.encode(molecule, self.encoding) to the input
        encoding = self.encode(molecule, enc= self.encoding)
        input_ = [np.append(encoding, row ) for row in input]


        # evaluate the kernel on the grid
        input_ = np.array(input_)
        predict = kernel.model.predict(input_)
        Z = predict[0].reshape(X.shape)
        err = predict[1].reshape(X.shape)
        return {"Z": Z, "err": err, "X": X, "Y": Y, "x_vec": x_vec, "y_vec": y_vec}


### ------------------------------ PLOTTING ------------------------------ ###


    def plot_data(self,
                    var1=None, var2 = None, var3 = None,
                    kernel = None,
                    molecule = "all",
                    library = "plotly",
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
        
        fig.update_layout(scene = dict(xaxis_title=var1+ "[10^4]", 
                                yaxis_title=var2 ,
                                zaxis_title="Peak Position [nm]",
                                ),
                                )

        # plot model over the parameter space
        if kernel is not None:
            
            # evaluate the kernel on a grid
            dict_ = self.evaluate_kernel(kernel, molecule)
            Z, err, X, Y, x_vec, y_vec = dict_["Z"], dict_["err"], dict_["X"], dict_["Y"], dict_["x_vec"], dict_["y_vec"]

            # add the surface plot of the kernel with a uniform color
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
        peak_pos = df[target]


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

            # save the data to a csv
            # df_Z = pd.DataFrame(data = Z, index = x_vec, columns = y_vec)
            # df_Z.to_csv(f"model_{molecule}_contour_data.csv")

            # colorbar with text size 12
            cbar = fig.colorbar(c, ax=ax, label = "PEAK POS",)
            cbar.ax.tick_params(labelsize=12,)
            cbar.set_label("peak position (nm)", fontsize = 12)


        # plot the data
        ax.scatter(x, y, c = peak_pos,  vmin = 400, vmax = 600, s = 80,
                    cmap = "gist_rainbow_r",
                    edgecolors='black', zorder = 3)
    
        
        # layout
        # ticks inside, label size 12
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True,)

        # set x and y range 
        ax.set_ylim(0., 1.)
        ax.set_xlim(0., 1.)

        # set labels
        ax.set_xlabel("AS/Pb ratio (10^4)", fontsize = 12)
        ax.set_ylabel("Cs/Pb ratio", fontsize = 12)
        ax.set_title(f"{molecule}", fontsize = 14)

        # plot as svg
        #plt.savefig(f"Contour_{molecule}.svg")

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
        #df = df[df["S/P"] == "P"]
        df = df[df["monodispersity"] == True]

        # remove everything with y=0
        df = df[df[var2] != 0]

        # get the data
        x = df[var1]
        y = df[var2]#*100 #1000
        color = df[color_var]
        sample_no = df["Sample No."]    

        peak_pos_eV = df["peak_pos_eV"]
        #suggestion = df["suggestion"]
        #suggestion = [1 if "L-" in str(s) else 0 for s in suggestion]
        LL = [1 if "LL" in str(s) else 0 for s in sample_no]
        #if color_var == "suggestion":
        #    color = suggestion

        # set color to black if the sample is LL
        #color = [0 if l == 1 else c for l, c in zip(LL, color)]




        """ basic scatter plot """
        fig, ax = plt.subplots(figsize = (4, 4))
        #fig, ax = plt.subplots(figsize = (6, 4))
        #fig, ax = plt.subplots(figsize = (10, 5.5))
        #fig, ax = plt.subplots(figsize = (4, 3))

        # custom color map# Choose original colormap
        original_cmap = plt.get_cmap('vanimo')

        # Get the original colormap colors
        n_colors = 256
        original_colors = original_cmap(np.linspace(0, 1, n_colors))
        half = n_colors // 2
        modified_colors = np.copy(original_colors)
        modified_colors[half:] = [0, 0, 0, 1]  # RGBA for black
        custom_cmap = ListedColormap(modified_colors)

        # cbar = plt.colorbar(ax.scatter(x, y,
        #                                 c = color, cmap= "bwr", alpha = 1, s = 70,vmin = 0, vmax = 1)) #vmin = 0, vmax = 0.3)) # vmin = nm_to_ev(400), vmax = nm_to_ev(600)))

        cmap = plt.get_cmap('gist_rainbow')
        cmap.set_under('k')
        cbar = plt.colorbar(ax.scatter(x, y, c = color, cmap= cmap, alpha = 1, s = 70, vmin = nm_to_ev(400), vmax = nm_to_ev(600)))
        # hide the colorbar
        cbar.remove()
        
        # label the points with the sample number
        # for i, txt in enumerate(sample_no):
        #     ax.annotate(txt, (np.array(x)[i]+0.001, np.array(y)[i]), fontsize = 8, color = "black")

        
        # settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,) # labeltop=True, labelbottom=False)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True, ) # labelleft=True, labelright=False)

        # set axis range
        # ax.set_ylim(-0.05, 1.05)
        # ax.set_xlim(-0.05, 1.05)

        # axis limits
        #ax.set_xlim(2.45, 2.65)

        # reverse x axis
        #ax.invert_xaxis()


        """ plot lines at ml boundaries """
        for ml in self.ml_dictionary.keys():
            peak_range = self.ml_dictionary[ml]
            peak_range = [ev_to_nm(peak_range[0]), ev_to_nm(peak_range[1])]
            #ax.axvline(x = peak_range[0], color = "black", linestyle = "dashed", linewidth = 1)
            #ax.axvline(x = peak_range[1], color = "black", linestyle = "dashed", linewidth = 1)

            #ax.axhline(y = peak_range[0], color = "black", linestyle = "dashed", linewidth = 1, alpha = 0.3)
            #ax.axhline(y = peak_range[1], color = "gray", linestyle = "dashed", linewidth = 1, alpha = 0.3)

            # fill between the lines
            #ax.fill_between([-0.05, 0.4], peak_range[0], peak_range[1], color = "gray", alpha = 0.1)

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
        
        #plt.show()


        """plot surface proportions"""
        # plt.tight_layout()
        # x = np.linspace(min(peak_pos_eV), max(peak_pos_eV), 300)
        # prop = [surface_proportion(x, "EV")*100 for x in x]
        # plt.plot(x, prop, "--", color = "black")

        plt.tight_layout()
        plt.show()
        #plt.savefig(f"data/{molecule_name}_As_Pb_peak_pos.png")

        # save the plot as svg
        #plt.savefig(f"plots/{var1}_{var2}.svg")

        # save data to csv
        #df = df[[var1, var2, "Sample No.", "molecule_name"]]
        #df.to_csv(f"plots/{var1}_{var2}.csv", mode='a', header=True, index=True)

        return fig, ax


    def plot_correlation(self, selection_dataframe = None) -> None:

        """
            Plots the correlation matrix of the data 
            together with the target values
        """

        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # properties = ["Cs_Pb_ratio", "t_Rkt", "V (Cs-OA)","peak_pos", "fwhm", "polydispersity", "Pb/I", 
        #               "Centrifugation time [min]", "Centrifugation speed [rpm]", "c (Cs-OA)",]
        properties = ["peak_pos", "V_total", "n_Cs", "n_As", "n_Pb", "Cs_Pb_ratio", "AS_Pb_ratio", "AS_Cs_ratio", ] #"relative polarity (-)","dielectric constant (-)","dipole moment (D)","Hansen parameter hydrogen bonding (MPa)1/2","Gutman donor number (kcal/mol)"]
        df = df[properties]


        # plot the correlation matrix
        corr = df.corr()
        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="bwr", vmin=-1, vmax=1)

        # save correlation matrix as csv
        corr.to_csv("plots/S_correlation_matrix.csv")


        # write the values in the matrix
        for i in range(len(properties)):
            for j in range(len(properties)):
                text = ax.text(j, i, round(corr.iloc[i, j], 2),
                                ha="center", va="center", color="black")


        # mark the peak_pos with a black border along the row
        for i in range(len(properties)):
            if properties[i] == "peak_pos":
                ax.axhline(i+0.5, color = "black", linewidth = 2)
                ax.axhline(i-0.5, color = "black", linewidth = 2)


        ax.set_xticks(np.arange(len(properties)))
        ax.set_yticks(np.arange(len(properties)))
        ax.set_xticklabels(properties, rotation = 45)
        ax.set_yticklabels(properties)


        plt.colorbar(im)
        plt.show()

