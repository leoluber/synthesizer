""" Collection of functions for vizualizing the data in the 
    data_processed.csv file
"""
    # <github.com/leoluber>



import matplotlib.pyplot as plt
import plotly
import pandas as pd
import numpy as np
import os

# custom
from package.src.Datastructure import Datastructure
from package.src.helpers import find_lowest, surface_proportion, ev_to_nm



# Plotter builds on the Datastructure class
class Plotter(Datastructure):

    """ Plotter class
    
    General porpuse class for plotting the data in the data/processed/data_processed.csv file

    """

    def __init__(self,
                 processed_file_path,
                 ):
        
        # read the data
        self.data_frame = pd.read_csv(processed_file_path, header=0, sep=";")

        # paths
        self.data_path_raw =            "data/raw/"
        self.geometry_path=             self.data_path_raw + "molecule_encoding.json"
        self.molecule_dictionary_path = self.data_path_raw + "molecule_dictionary.json"
        self.ml_dictionary_path =       self.data_path_raw + "ml_dictionary.json"

        # get encodings
        self.molecule_dictionary, self.ml_dictionary, self.molecule_geometry = self.get_dictionaries()





    def plot_data(self,
                    var1=None, var2 = None, var3 = None,
                    kernel = None,
                    molecule = "all",
                    library = "plotly",
                    selection_dataframe = None) -> None:

        """
            Simple 3D Scatter plot of the data in parameter space, for visualization;
            kernel can be plotted as well
        """

        # choose the data frame
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # select the data for the correct molecule
        if molecule != "all":
            df = df[df["molecule_name"] == molecule]

        
        # different colors for the artificial data
        artificial = df[df["artificial"] == True]
        df = df[df["artificial"] == False]


        # get the data
        var_1 = df[var1]
        var_2 = df[var2]
        var_3 = df[var3]
        peak  = df["peak_pos"]
        sample_no = df["Sample No."]

        # get the artificial data
        art_1 = artificial[var1]
        art_2 = artificial[var2]
        art_3 = artificial[var3]


        # plot data with plotly
        if library == "plotly":
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects
                            .Scatter3d(x=var_1, y=var_2, z=var_3, mode='markers', 
                                    marker=dict(size=13, opacity=0.7, color=peak,),
                                    text = sample_no,
                                    ))
            
            fig.update_traces(marker=dict(cmin=380, cmax=650, colorbar=dict(title='PEAK POS'), 
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
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel("Peak Position [nm]")


        # plot model over the parameter space
        if kernel is not None:
            y_vec = np.linspace(0, 1, 100)
            x_vec = np.linspace(0, 0.8, 100)
            X, Y  = np.meshgrid(x_vec, y_vec)
            input = np.c_[X.ravel(), Y.ravel()]
            
            # add self.encode(molecule, self.encoding) to the input
            encoding = self.encode(molecule)
            input = [np.append(encoding, row ) for row in input]

            input = np.array(input)
            Z = kernel.model.predict(input)[0].reshape(X.shape)
            err = kernel.model.predict(input)[1].reshape(X.shape)


            # write X, Y, Z to a csv with pandas
            #df = pd.DataFrame(data = Z, index = x_vec, columns = y_vec)
            #df.to_csv(f"model_{molecule}.csv")



            # add the surface plot of the kernel with a uniform color
            if library == "plotly":
                fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=0.8, colorscale='greys', cmin = 200, cmax = 900))
                #fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=1, colorscale='Viridis', cmin = 430, cmax = 540))
            # elif library == "matplotlib":
            #     ax.plot_surface(X, Y, Z, alpha=0.7, color = "gray", lw=0.5, rstride=8, cstride=8,)
            #     ax.contourf(X, Y, Z, zdir='z', offset=420, cmap='gist_rainbow_r', alpha=0.8, vmin = 410, vmax = 600, levels = 20)
                
            #     ax.scatter(As_Pb, Cs_Pb, peak, c = peak, cmap = "gist_rainbow_r", 
            #                edgecolors='black', vmin = 410, vmax = 600, s = 100, alpha=1)
            #     ax.scatter(As_Pb_base, Cs_Pb_base,  peak_base, c = "black", edgecolors='black', s = 100, alpha=1)

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
    

        ### -- a contour plot of the kernel -- ###
        y_vec   = np.linspace(0, 1, 100)
        x_vec   = np.linspace(0, 1, 100)
        X, Y    = np.meshgrid(x_vec, y_vec)
        input   = np.c_[X.ravel(), Y.ravel()]
        
        # evaluate the kernel on the grid
        encoding = self.encode(molecule)
        input = np.array([np.append(encoding, row ) for row in input])

        if kernel is not None:
            Z = kernel.model.predict(input)[0].reshape(X.shape)

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
        fig.write_image(f"plots/Contour_Monodisp_{molecule}.svg")
        fig.write_image(f"plots/Contour_Monodisp_{molecule}.png")

        fig.show()



    def plot_2D_contour_old(self, var1 = None, var2 = None, 
                            kernel = None, molecule = None,
                            selection_dataframe = None) -> None:

        """
            2D contour plot of the data in parameter space, for visualization 
            purposes color coded by the target value (PEAK_POS)
            
            >>> ADDED: area calculation below 464nm
        """

        # choose the data frame
        if selection_dataframe is not None:
            df = selection_dataframe
        else:
            df = self.data_frame

        # select the data for the correct molecule
        df = df[df["molecule_name"] == molecule]

        # get the data 
        x = df["AS_Pb_ratio"]
        y = df["Cs_Pb_ratio"]
        peak_pos = df["peak_pos"]


        # a contour plot of the kernel
        fig, ax = plt.subplots()
        y_vec   = np.linspace(0, 1, 100)
        x_vec   = np.linspace(0, 1, 100)
        X, Y    = np.meshgrid(x_vec, y_vec)
        input   = np.c_[X.ravel(), Y.ravel()]

        
        # evaluate the kernel on the grid
        encoding = self.encode(molecule,)
        input = np.array([np.append(encoding, row ) for row in input])


        # evaluate the kernel on the grid
        if kernel is not None:
            Z = kernel.model.predict(input)[0].reshape(X.shape)
            c = ax.contourf(X, Y, Z, 30, cmap='gist_rainbow_r', vmin = 400, vmax = 600, zorder = 1)

            # add black lines for the contour
            ax.contour(X, Y, Z, 30, colors='black', linewidths=0.5, zorder = 2)
            
            # colorbar with text size 12
            cbar = fig.colorbar(c, ax=ax, label = "PEAK POS",)
            cbar.ax.tick_params(labelsize=12,)
            cbar.set_label("PEAK POS", fontsize = 12)


        # plot the data
        ax.scatter(x, y, c = peak_pos, s = 80, vmin = 400, vmax = 600, 
                    cmap = "gist_rainbow_r",
                    edgecolors='black', zorder = 3)
        
        # layout
        # ticks inside, label size 12
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True,)

        # set x and y range to -0.1, 1.1
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        # set labels
        ax.set_xlabel("AS/Pb Ratio (10^4)", fontsize = 12)
        ax.set_ylabel("Cs/Pb Ratio", fontsize = 12)

        # save the plot as svg
        plt.savefig(f"plots/Contour_{molecule}.svg")
        
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

        # get the data
        x = [data[var1] for data in df]
        y = [data[var2] for data in df]
        color = [data[color_var] for data in df]
        peak_pos = [data["peak_pos_eV"] for data in df]

        # other
        suggestion = [data["suggestion"] for data in df]


        
        # lowest values for each peak position
        # TODO: replace in new version
        # lowest_x, lowest_y = find_lowest(data_objects=data_objects)
        # lowest_y = [y * 1000 for y in lowest_y]
        


        """ basic scatter plot """
        fig, ax = plt.subplots(figsize = (3.5, 4))
        sign = -1 if self.wavelength_unit == "EV" else 1

        cbar = plt.colorbar(ax.scatter(x, y,
                                        c = color, cmap= "bwr_r", alpha = 1, s = 70,vmin = 0, vmax = 0.2)) # vmin = nm_to_ev(400), vmax = nm_to_ev(600)))
        
        # cbar = plt.colorbar(ax.scatter(x, y
        #                                 c = color, cmap= "gist_rainbow_r", alpha = 1, s = 70, vmin = 400, vmax = 600))
        
        # label the points with the sample number
        # for i, txt in enumerate(sample_no):
        #     ax.annotate(txt, (c_Perovskite[i], target[i]), fontsize = 8, color = "black")


        # scatters lower limits
        #ax.scatter(lowest_x, lowest_y, c = "red", s = 50, label = "lowest")
        #cbar.remove()
        
        # settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12, top=True, bottom=True,) # labeltop=True, labelbottom=False)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12, right=True, left=True, ) #labelleft=True, labelright=False)

        # set axis range
        #ax.set_ylim(0, 1)
        #ax.set_xlim(0, 1)


        """ plot lines at ml boundaries """
        for ml in self.ml_dictionary.keys():
            peak_range = self.ml_dictionary[ml]
            peak_range = [ev_to_nm(peak_range[0]), ev_to_nm(peak_range[1])] if self.wavelength_unit == "EV" else peak_range
            #ax.axvline(x = peak_range[0], color = "black", linestyle = "dashed", linewidth = 1)
            #ax.axvline(x = peak_range[1], color = "black", linestyle = "dashed", linewidth = 1)

            #ax.axhline(y = peak_range[0], color = "black", linestyle = "dashed", linewidth = 1, alpha = 0.3)
            #ax.axhline(y = peak_range[1], color = "gray", linestyle = "dashed", linewidth = 1, alpha = 0.3)

            # fill between the lines
            #ax.fill_between([-0.05, 0.4], peak_range[0], peak_range[1], color = "gray", alpha = 0.1)

        
        #plt.show()


        """plot surface proportions"""
        plt.tight_layout()
        x = np.linspace(min(peak_pos), max(peak_pos), 300)
        prop = [surface_proportion(x, "EV") for x in x]
        plt.plot(x, prop, "--", color = "black")


        # set axis range
        #ax.set_xlim(2.650, 2.750)
        #ax.set_ylim(55, 135)

        plt.tight_layout()
        plt.show()
        fig.show()
        #plt.savefig(f"data/{molecule_name}_As_Pb_peak_pos.png")


        # save as csv
        #df = pd.DataFrame({"AS_Pb_ratio": AS_Pb_ratio, "Cs_Pb_ratio": Cs_Pb_ratio, "peak_pos": peak_pos, "monodispersity": monodispersity, "fwhm": fwhm, "suggestion": suggestion, "molecule_name": molecule_name})
        #df.to_csv("data/parameters.csv", index = False)

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
        encoding = self.encode(molecule, self.encoding)
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

        properties = ["Cs_Pb_ratio", "t_Rkt", "V (Cs-OA)","peak_pos", "fwhm", "polydispersity", "Pb/I", 
                      "Centrifugation time [min]", "Centrifugation speed [rpm]", "c (Cs-OA)",]
        df = df[properties]
        print(df["t_Rkt"])

        # plot the correlation matrix
        corr = df.corr()
        fig, ax = plt.subplots()
        # center the colorbar at 0
        im = ax.imshow(corr, cmap="bwr", vmin=-1, vmax=1)

        # mark the peak_pos with a black border along the row
        for i in range(len(properties)):
            if properties[i] == "peak_pos":
                ax.axhline(i+0.5, color = "black", linewidth = 2)
                ax.axhline(i-0.5, color = "black", linewidth = 2)

        # --> the "+3" is used for the additional plqy, fwhm, peak_pos
        ax.set_xticks(np.arange(len(properties)))
        ax.set_yticks(np.arange(len(properties)))
        ax.set_xticklabels(properties, rotation = 45)
        ax.set_yticklabels(properties)


        plt.colorbar(im)
        plt.show()



    def plot_ternary(self, data_objects, kernel= None) -> None:

        """
            Ternary plot of the data in parameter space, for visualization
        """

        # amount of each substance
        Cs = np.array([data["amount_substance"]["Cs"] for data in data_objects])
        Pb = np.array([data["amount_substance"]["Pb"]/4 for data in data_objects])
        As = np.array([data["amount_substance"]["As"] for data in data_objects])
        total = Cs + Pb + As

        # normalize the data
        Cs = Cs / total
        Pb = Pb / total
        As = As / total

        #TODO: fix

        # plot the data
        fig = go.Figure(go.Scatterternary(a=Cs, b=Pb, c=As, 
                        mode='markers', 
                        marker=dict(size=10, opacity=0.7, cmin=400, cmax=0, 
                                    color = [data["peak_pos"] for data in data_objects],
                                    colorscale='rainbow',),
                        ))
        fig.update_layout({
            'ternary': {'sum': 1, 'aaxis': {'title': 'Cs'}, 'baxis': {'title': 'Pb'}, 'caxis': {'title': 'As'}},
            'annotations': [{'showarrow': False, 'text': f'{self.flags["molecule"]}', 
                                'x': 0.5, 'y': -0.2, 'font': {'size': 16}}],
            'width': 400,
            'height': 400,
                        })

        fig.show()

        # save as svg
        fig.write_image(f"plots/Ternary_{self.flags['molecule']}.svg")



