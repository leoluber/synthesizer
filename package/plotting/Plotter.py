""" 
    Project:     synthesizer
    File:        Datastructure.py
    Description: Defines the Plotter class (child class of Datastructure) 
                 for visualizing the processed data
    Author:      << github.com/leoluber >> 
    License:     MIT
"""





import matplotlib.pyplot as plt
import plotly
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# custom imports
from src.Datastructure import Datastructure
from src.helpers import surface_proportion, ev_to_nm, nm_to_ev






class Plotter(Datastructure):

    """ General porpuse class for plotting the processed data in various ways

    NOTE: DON'T publish this class, it's just for internal use
    
    (...)
    
    """


    def __init__(self,
                 processed_file_path,
                 encoding = "combined",
                 selection_dataframe = None,
                 ):
        
        # No super().__init__() snince we don't actually need to
        # initialize the Datastructure class, we just want to inherit the methods

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
        self.molecule_dictionary, self.ml_dictionary, self.encoding_dictionary, self.molecule_geometry = self.get_dictionaries()

        # molecule attributes
        self.global_attributes_df =  pd.read_csv(self.global_attributes_path, 
                                                 delimiter= ';', header= 0)




### ---------------------------- INIT RELATED ---------------------------- ###

    def evaluate_kernel(self, kernel, molecule) -> dict:

        """ Evaluates the kernel on a grid for visualization purposes
        
        (...)

        """

        # set the grid (and define bounds of parameter space)
        y_vec = np.linspace(0, 1, 100)
        x_vec = np.linspace(0, 1, 100)
        X, Y  = np.meshgrid(x_vec, y_vec)
        input = np.c_[X.ravel(), Y.ravel()]
        
        # add self.encode(molecule, self.encoding) to the input
        encoding = self.encode(molecule, enc= self.encoding)
        #encoding = self.encode(molecule)
        input_ = [np.append(encoding, row ) for row in input]


        # evaluate the kernel on the grid
        input_ = np.array(input_)
        print(input_[0])
        Z = kernel.model.predict(input_)[0].reshape(X.shape)
        err = kernel.model.predict(input_)[1].reshape(X.shape)

        
        # append input to data.csv
        # df = pd.DataFrame(data = input, columns = ["AS_Pb_ratio", "Cs_Pb_ratio",])
        # df["peak_pos"] = Z.ravel()
        # df["Sample No."] = np.array(["synth" for i in range(len(df))])
        # df["molecule_name"] = np.array([molecule for i in range(len(df))])

        # save the data
        #df.to_csv("data.csv", mode='a', header=False)


        return {"Z": Z, "err": err, "X": X, "Y": Y, "x_vec": x_vec, "y_vec": y_vec}


    def screen_kernel(self, kernel, molecule, Cs_Pb = 0.3, As_Pb = None) -> dict:

        """ Evaluates the kernel on a line for screening purposes
        
        (...)

        """
        
        fig, ax = plt.subplots()

        if (Cs_Pb is None and As_Pb is None) or (Cs_Pb is not None and As_Pb is not None):
            raise ValueError("Please provide at exactly one parameter")
        
        if Cs_Pb is not None:
            As_Pb_values = np.linspace(0, 1, 100)
            inputs = np.array([np.append(self.encode(molecule), [As_Pb, Cs_Pb, ]) for As_Pb in As_Pb_values])
            Z = kernel.model.predict(inputs)[0]

            # calc V_As
            V_As = As_Pb_values * (0.0066 * 1000 * 10000) /(11.29)
            ax.plot(V_As, Z, color = "black", linestyle = "dashed", linewidth = 2)

        elif As_Pb is not None:
            Cs_Pb_values = np.linspace(0, 1, 100)
            inputs = np.array([np.append(self.encode(molecule), [As_Pb, Cs_Pb, ]) for Cs_Pb in Cs_Pb_values])
            Z = kernel.model.predict(inputs)[0]
            ax.plot(Cs_Pb_values, Z, color = "black", linestyle = "dashed", linewidth = 2)
            

        #plot
        ax.set_xlabel("As")
        ax.set_ylabel("Peak Position [nm]")
        plt.show()

        


### ------------------------------ PLOTTING ------------------------------ ###



    def plot_data(self,
                    var1=None, var2 = None, var3 = None,
                    kernel = None,
                    molecule = "all",
                    library = "plotly",
                    selection_dataframe = None) -> None:


        """ Simple 3D Scatter plot of the data in parameter space, for visualization;
            kernel can be plotted as well

        (...)

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

        # get all polydisperse data
        df_poly = df[df["monodispersity"] ==0]
        print(df)
        df_notpoly = df[df["monodispersity"] ==1]
        print(df_notpoly)
        if len(df_poly) > 0:
            for i in range(10):
                print("ATTENTION: POLYDISPERSE DATA")


        # get the data
        var_1 = df[var1]
        var_2 = df[var2]
        var_3 = df[var3]
        peak  = df["peak_pos"]
        sample_no = df["Sample No."]
        fwhm = df["fwhm"]
        #suggestion = df["suggestion"]
        info = [f"No: {str(no)}  FWHM: {str(round(fwhm, 2))}" for no, fwhm in zip(sample_no, fwhm)]

        # get the artificial data
        art_1 = artificial[var1][:5]
        art_2 = artificial[var2][:5]
        art_3 = artificial[var3][:5]

        # plot data with plotly
        if library == "plotly":
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects
                            .Scatter3d(x=var_1, y=var_2, z=var_3, mode='markers', 
                                    marker=dict(size=13, opacity=0.7, color=peak,),
                                    text = info,
                                    ))
            
            fig.update_traces(marker=dict(cmin=400, cmax=600, colorbar=dict(title='PEAK POS'), 
                                    colorscale='rainbow', color=peak, showscale=True, opacity=1),
                                textposition='top center')
            
            # fig.update_traces(marker=dict(cmin=0, cmax=80, colorbar=dict(title='PEAK POS'), 
            #                         colorscale='viridis', color=peak, showscale=True, opacity=1),
            #                  textposition='top center')

            fig.add_trace(plotly.graph_objects
                            .Scatter3d(x=art_1, y=art_2, z=art_3, mode='markers', 
                                        marker=dict(size=13, opacity=1, color="black"),
                                        ))
            
            fig.update_layout(scene = dict(xaxis_title=var1+ "[10^4]", 
                                    yaxis_title=var2 ,
                                    #zaxis_title="Peak Position [nm]",
                                    ),
                                    )

        elif library == "matplotlib":
            fig = plt.figure(figsize=(4.5, 4.5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("As/Pb ratio (10^4)")
            ax.set_ylabel("Cs/Pb ratio")
            ax.set_zlabel("peak position (nm)")



        # plot model over the parameter space
        if kernel is not None:

            # screen
            #self.screen_kernel(kernel, molecule, Cs_Pb = 0.3)
            
            # evaluate the kernel on a grid
            dict_ = self.evaluate_kernel(kernel, molecule)
            Z, err, X, Y, x_vec, y_vec = dict_["Z"], dict_["err"], dict_["X"], dict_["Y"], dict_["x_vec"], dict_["y_vec"]
            
            # write X, Y, Z to a csv with pandas
            df_ = pd.DataFrame(data = Z, index = x_vec, columns = y_vec)
            df_.to_csv(f"model_{molecule}_S.csv")


            # add the surface plot of the kernel with a uniform color
            if library == "plotly":
                fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=0.8, colorscale='greys', cmin = 400, cmax = 600))
                #fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=1, colorscale='Viridis', cmin = 430, cmax = 540))
            
            elif library == "matplotlib":
                ax.plot_surface(X, Y, Z, alpha=0.7, color = "gray", lw=0.5, rstride=5, cstride=5,)
                ax.contourf(X, Y, Z, zdir='z', offset=420, cmap='gist_rainbow_r', alpha=0.8, vmin = 400, vmax = 600, levels = 10)
                
                ax.scatter(var_1, var_2, var_3, c = peak, cmap = "gist_rainbow_r", 
                           edgecolors='black', vmin = 400, vmax = 600, s = 50, alpha=1)
                ax.scatter(art_1, art_2, art_3, c = "black", edgecolors='black', s = 50, alpha=1)

            # add confidence intervals
            # if model == "GP":
            #     if library == "plotly":
            #         #fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z + 1.96 * err, opacity=0.5, colorscale='gray'))
            #         # fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z - 1.96 * err, opacity=0.5, colorscale='gray'))
            #         fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z= err/10 + 410, opacity=0.5, colorscale='gray'))
            #     # elif library == "matplotlib":
            #         # ax.plot_surface(X, Y, Z + 1.96 * err, alpha=0.5, color = "gray")
            #         # ax.plot_surface(X, Y, Z - 1.96 * err, alpha=0.5, color = "gray")


        # save the plot as interactive html
        if library == "plotly":
            #fig.write_html(f"plots/{molecule}.html")
            fig.show()
        
        elif library == "matplotlib":
            plt.show()

        # plot toluene baseline
        fig = plt.figure(figsize=(2.5, 3.5))
        data = df[df["V (antisolvent)"] == 0]
        peak_pos = data["peak_pos"]
        cs_pb_ratio = data["Cs_Pb_ratio"]
        model = Z[:, 0]
        plt.scatter(cs_pb_ratio, peak_pos, cmap = "gist_rainbow_r", vmin=400, vmax = 600, c = peak_pos)
        plt.plot(x_vec, model, "--", color = "black", linewidth = 1, label = "Model")

        # # add grey lines for the ML boundaries
        # for ml in self.ml_dictionary.keys():
        #     # if ml in ["7", "8",]:
        #     #     continue
        #     peak_range = self.ml_dictionary[ml]
        #     peak_range = [peak_range[0], peak_range[1]]
        #     plt.fill_between([-0.05, 1.05], peak_range[0], peak_range[1], color = "gray", alpha = 0.1)

        # data = df[df["Cs_Pb_ratio"] == 0.20]
        # peak_pos = data["peak_pos"]
        # as_pb_ratio = data["AS_Pb_ratio"]
        # model = Z[20, :]
        # plt.scatter(as_pb_ratio, peak_pos, cmap = "gist_rainbow_r", vmin=400, vmax = 600, c = peak_pos)
        # plt.plot(x_vec[0:40], model[0:40], "--", color = "black", linewidth = 1, label = "Model")


        # # layout
        #plt.xlabel("Cs/Pb ratio")
        plt.xlabel("As/Pb ratio (10^4)")
        plt.ylabel("peak position (nm)")
        plt.xlim(-0.05, 0.70)
        plt.ylim(430, 520)
        plt.gca().xaxis.set_tick_params(direction='in', which='both', top=True, bottom=True,)
        plt.gca().yaxis.set_tick_params(direction='in', which='both', right=True, left=True,)
        plt.tight_layout()
        plt.show()

        return fig
        


    def plot_2D_contour(self, var1 = None, var2 = None, 
                    molecule = None, kernel = None,
                    selection_dataframe = None) -> None:

        """
            2D contour plot of the data in parameter space, for visualization 
            purposes color coded by the target value (PEAK_POS)
        """

        # choose the data frame
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame


        # select the data for the correct molecule
        df = df[df["molecule_name"] == molecule]

        # remove the baseline data
        df = df[df["baseline"] == False]


        # get the data
        x = df[var1]
        y = df[var2]


        if kernel is not None:
            
            # evaluate the kernel on a grid
            dict_ = self.evaluate_kernel(kernel, molecule)
            Z, err, X, Y, x_vec, y_vec = dict_["Z"], dict_["err"], dict_["X"], dict_["Y"], dict_["x_vec"], dict_["y_vec"]

            # contour plot with plotly, figure size is set
            fig = plotly.graph_objects.Figure( layout = dict(width = 580, height = 500))
            fig.add_trace(plotly.graph_objects.Contour(x=x_vec, y=y_vec, z=Z, contours = dict(start=400, 
                                                                                                end=600, 
                                                                                                size=3,
                                                                                                coloring='lines',
                                                                                                ), 
                                                        line = dict(width=1),
                                                        colorscale= [[0, 'rgb(0, 0, 0)'], [1, 'rgb(0, 0, 0)']],
                                                        showscale=False,
                                                        ))
            
            fig.update_layout(scene = dict(xaxis_title=var1, yaxis_title=var2, zaxis_title="PEAK POS")) #, coloraxis_showscale=False)


        # plot the data
        c = df["monodispersity"]
        fig.add_trace(plotly.graph_objects.Scatter(x=x, y=y, mode='markers',
                                                    marker=dict(size=16, opacity=1, color = c, 
                                                                colorscale='hot', cmin=0, cmax=50,
                                                                showscale=True, colorbar=dict(title='FWHM [mev]]',
                                                                tickfont=dict(size=16),),
                                                                line=dict(width=1, color='Black'),
                                                                ),
                                                    ))
        
        # layout: no ticks, no grid, no lines, no labels
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title_font = dict(size = 16))
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title_font = dict(size = 16))
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)


        # save the plot as svg and png
        #fig.write_image(f"plots/Contour_Monodisp_{molecule}.svg")
        #fig.write_image(f"plots/Contour_Monodisp_{molecule}.png")

        fig.show()



    def plot_2D_contour_old(self, var1 = None, var2 = None, 
                            kernel = None, molecule = None,
                            selection_dataframe = None) -> None:


        """ 2D contour plot of the data in parameter space
            
            >>> ADDED: area calculation below 464nm
            (return)

            (...)

        """

        # choose the data frame
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # select the data for the correct molecule
        df = df[df["molecule_name"] == molecule]

        # remove the baseline data
        df = df[df["baseline"] == False]

        df_poly = df[df["monodispersity"] == 0]

        # get the data 
        x = df["AS_Pb_ratio"]
        x_poly = df_poly["AS_Pb_ratio"]
        y = df["Cs_Pb_ratio"]
        y_poly = df_poly["Cs_Pb_ratio"]
        peak_pos = df["peak_pos"]
        peak_pos_poly = df_poly["peak_pos"]


        # a contour plot of the kernel
        fig, ax = plt.subplots(figsize=(5, 4))


        # evaluate the kernel on the grid
        if kernel is not None:

            # evaluate the kernel on a grid
            dict_ = self.evaluate_kernel(kernel, molecule)
            Z, err, X, Y, x_vec, y_vec = dict_["Z"], dict_["err"], dict_["X"], dict_["Y"], dict_["x_vec"], dict_["y_vec"]

            c = ax.contourf(X, Y, Z, 30, cmap='gist_rainbow_r', vmin = 400, vmax = 600, zorder = 1)

            # add black lines for the contour
            ax.contour(X, Y, Z, 30, colors='black', linewidths=0.5, zorder = 2)

            # save the data to a csv
            df_Z = pd.DataFrame(data = Z, index = x_vec, columns = y_vec)
            df_Z.to_csv(f"model_{molecule}_contour_data.csv")


            # colorbar with text size 12
            cbar = fig.colorbar(c, ax=ax, label = "PEAK POS",)
            cbar.ax.tick_params(labelsize=12,)
            cbar.set_label("peak position (nm)", fontsize = 12)


        # plot the data
        ax.scatter(x, y, c = peak_pos, s = 50, vmin = 400, vmax = 600, 
                    cmap = "gist_rainbow_r",
                    edgecolors='black', zorder = 3)
        ax.scatter(x_poly, y_poly, c = "white", s = 50,
                    edgecolors='black', zorder = 3)
        
        # scatter polydisperse data in white
        print(df)
        x_p = df[df["monodispersity"] == False]["AS_Pb_ratio"]
        y_p = df[df["monodispersity"] == False]["Cs_Pb_ratio"]
        ax.scatter(x_p, y_p, c = "white", s = 70, edgecolors='black', zorder = 3)
        
        # layout
        # ticks inside, label size 12
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True,)

        # set x and y range to -0.1, 1.1
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # set labels
        ax.set_xlabel("AS/Pb ratio (10^4)", fontsize = 12)
        ax.set_ylabel("Cs/Pb ratio", fontsize = 12)

        # save the plot as svg
        plt.savefig(f"Contour_{molecule}.svg")
        plt.show()


        # calculate the area below 464nm (3ML or less)
        area = 0
        for nm in Z:
            for val in nm:
                if val <= 464:
                    area += 1

        return area
    


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
        y = df[var2]*1000
        color = df[color_var]
        sample_no = df["Sample No."]    

        peak_pos_eV = df["peak_pos_eV"]
        suggestion = df["suggestion"]
        suggestion = [1 if "L-" in str(s) else 0 for s in suggestion]
        LL = [1 if "LL" in str(s) else 0 for s in sample_no]
        if color_var == "suggestion":
            color = suggestion

        # set color to black if the sample is LL
        color = [0 if l == 1 else c for l, c in zip(LL, color)]




        """ basic scatter plot """
        #fig, ax = plt.subplots(figsize = (3, 4))
        fig, ax = plt.subplots(figsize = (6, 4))
        #fig, ax = plt.subplots(figsize = (10, 5.5))
        #fig, ax = plt.subplots(figsize = (4, 3))

        # cbar = plt.colorbar(ax.scatter(x, y,
        #                                 c = color, cmap= "bwr_r", alpha = 1, s = 70,vmin = 0, vmax = 0.2)) # vmin = nm_to_ev(400), vmax = nm_to_ev(600)))

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
        #ax.set_ylim(0, 1)
        #ax.set_xlim(0, 1)

        # axis limits
        #ax.set_xlim(2.45, 2.65)

        # reverse x axis
        ax.invert_xaxis()


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

        
        #plt.show()


        """plot surface proportions"""
        # plt.tight_layout()
        # x = np.linspace(min(peak_pos_eV), max(peak_pos_eV), 300)
        # prop = [surface_proportion(x, "EV") for x in x]
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





#### TODO: fix everything below here in new version

    def plot_screening(self, data_objects, model = None) -> None:

        """
            Plots arbitrary parameters for different visualization purposes
            -> not pretty, but useful for quick checks
        """


        Cs_Pb_ratio =     	[data["Cs_Pb_ratio"] for data in data_objects]
        AS_Pb_ratio =       [data["AS_Pb_ratio"] for data in data_objects]
        target =            [data["y"]                   for data in data_objects]
        peak_pos =          [data["peak_pos"]            for data in data_objects]
        molecule = data_objects[0]["molecule_name"]

        if len(Cs_Pb_ratio) == 0:
            raise ValueError(f"Not enough data found")


        # evaluate the kernel on the line
        fixed = Cs_Pb_ratio[0]
        encoding = self.encode(molecule,)
        y_vec = np.linspace(0, 1, 100)
        input = np.array([np.append(encoding, [y, fixed]) for y in y_vec])
        output = model.model.predict(input)[0]

        
        """ basic scatter plot """
        fig, ax = plt.subplots(figsize = (4, 4))
        sign = -1 if self.wavelength_unit == "EV" else 1
        cbar = plt.colorbar(ax.scatter(AS_Pb_ratio, peak_pos,
                                        c = peak_pos, cmap= "gist_rainbow_r", alpha = 1, s = 80, vmin = 400, vmax = 600)) #vmin = nm_to_ev(400), vmax = nm_to_ev(600)))

        """ plot the kernel """
        ax.plot(y_vec, output, color = "black", linestyle = "dashed", linewidth = 2)


        cbar.remove()
        
        # settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True, ) # labeltop=True, bottom=False, labelbottom=False)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True, ) #labelleft=True, labelright=False)



        plt.show()
        fig.show()


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




    def plot_ternary(self, selection_dataframe = None, molecule = None) -> None:

        """
            Ternary plot of the data in parameter space, for visualization

            TODO: solve bug with these plots
        """

        # get the data
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # get correct columns
        n_Cs = df["n_Cs"][df["molecule_name"] == molecule]
        n_As = df["n_As"][df["molecule_name"] == molecule]
        n_Pb = df["n_Pb"][df["molecule_name"] == molecule]


        Cs = n_Cs / (n_Cs + n_As + n_Pb)
        As = n_As / (n_Cs + n_As + n_Pb)
        Pb = n_Pb / (n_Cs + n_As + n_Pb)


        # plot the data
        fig = go.Figure(go.Scatterternary(a=Cs, b=Pb, c=As, 
                        mode='markers', 
                        marker=dict(size=10, opacity=0.7, cmin=400, cmax=0, 
                                    color = df["peak_pos"],
                                    colorscale='rainbow',),
                        ))
        fig.update_layout({
            'ternary': {'sum': 1, 'aaxis': {'title': 'Cs'}, 'baxis': {'title': 'Pb'}, 'caxis': {'title': 'As'}},
            'annotations': [{'showarrow': False, 'text': f'{molecule}', 
                                'x': 0.5, 'y': -0.2, 'font': {'size': 16}}],
            'width': 400,
            'height': 400,
                        })

        fig.show()

        # save as svg
        fig.write_image(f"plots/Ternary_{molecule}.svg")
