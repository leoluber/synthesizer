import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly

# custom
from helpers import *




"""
    General purpose data structure that reads in the synthesis data, spectral information 
    and the global attributes of the molecules and creates a list of dict.objects; can be 
    easily extended to include more global attributes or other target values.

    DATA: 
    - all data should be at "/data/" relative to the current working directory and contain:
        - a folder "spectrum" with the spectral data in .txt files (csv)
        - AntisolventProperties.csv with the global attributes of the molecules; names for 
          properties to be found in the global_attribute_selection list
        - a .csv file with the synthesis data (synthesis_file_path) containing the sample 
          numbers, molecule names and synthesis parameters

"""


class Datastructure:

    def __init__(self,
                 synthesis_file_path,                          # path to the synthesis data file
                 target =              "FWHM",                 # Options: PEAK_POS, PLQY, FWHM, NPL_TYPE, NR
                 output_format =       "LIST",                 # LIST or TENSOR (for final output format of array-like data)
                 wavelength_filter =    [400, 600],            # filter for the spectrum, always in "NM"
                 exclude_no_star =      False,                 # exclude samples with bad PLQY measurements
                 exclude_PLQY =         False,
                 wavelength_unit =     "EV",                   # "EV" or "NM
                 molecule =            "all",                  # "all" or specific molecule
                 normalization =        True,                  # normalize the data if "True"
                 monodispersity_only =  False,                 # only includes samples with monodispersity flag if "True"
                 encoding =            "one_hot",              # "one_hot" or "geometry"
                 ):
        

        # main stettings
        self.target = target
        self.output_format = output_format
        self.encoding = encoding
        self.wavelength_unit = wavelength_unit
        self.normalization = normalization

        # flags
        self.exclude_no_star = exclude_no_star
        self.exclude_PLQY = exclude_PLQY
        self.molecule = molecule
        self.monodispersity_only = monodispersity_only

        # directories  (define data related directories here)
        self.current_path =  os.getcwd()
        self.data_path =                self.current_path + "/data/"
        self.synthesis_file_path =      self.data_path + synthesis_file_path
        self.global_attributes_path =   self.data_path + "AntisolventProperties.csv"
        self.spectrum_path =            self.data_path + "spectrum/"

        # data related local variables
        self.max_min = {"FWHM" : [0, 1000],}                                            # dictionary to store the max and min values of the parameters for denormalization
        self.data = []                                                                  # should not be accessed, use get_data() instead
        if wavelength_unit == "EV":
            self.wavelength_filter = [nm_to_ev(wavelength_filter[0]), 
                                      nm_to_ev(wavelength_filter[1])]
        else:
            self.wavelength_filter = wavelength_filter

        
        # parmeter selections (make sure this fits th csv headers in the data files); As_Pb_ratio and Pb_Cs_ratio are calculated during initialization

        # ------- ADJUSTABLE ------- #
        self.global_attribute_selection =       [] #"Hansen parameter hydrogen bonding (MPa)1/2", 'relative polarity (-)', "dielectric constant (-)",  "Hansen parameter hydrogen bonding (MPa)1/2", "dipole moment (D)", "Gutman donor number (kcal/mol)", "viscosity (mPa ?s)" ,]
        self.synthesis_training_selection =     [ "AS_Pb_ratio",] # "V (antisolvent)", "V (Cs-OA)", "V (PbBr2 prec.)", "c (PbBr2)", "Pb_Cs_ratio", ] #  "c (Cs-OA)", "c (OlAm)", "c (OA)"] #
        # -------------------------- #

        self.synthesis_selection =              [ "c (PbBr2)", "c (OlAm)", "c (OA)", "V (Cs-OA)", "c (Cs-OA)" ,"V (antisolvent)", "V (PbBr2 prec.)"]        
        self.total_training_parameter_selection = self.synthesis_training_selection + self.global_attribute_selection

        # read in the data
        self.synthesis_data = self.read_synthesis_data()                                # container of synthesis data arrays,           read in during initialization
        self.global_attributes_df = self.read_global_attributes_and_nomalize()          # dataframe of global attributes,               read in during initialization

        # dictionaries (will need to be adjusted for new molecules)
        self.molecule_dictionary, self.atom_to_num, self.num_to_atom, self.ml_dictionary, self.molecule_geometry = self.get_molecule_dictionary()

        # list of all relevant molecule names
        self.molecule_names = list(self.molecule_dictionary.values())



    def get_data(self):
        
        """
            - loop that iterates through the samples in the synthesis file, reads in all the extra information and 
              represents it as a Data object
            - synthesis data and the gloal attributes dataframe are read in during initialization 
            (accessed via self.synthesis_data["property_name"][index])
            - molecule specific data is read in during the loop
        """


        for index, sample_number in enumerate(self.synthesis_data["sample_numbers"]):
           
            # translates the data dependent molecule name to inherent module code (because I'm not a chemist and this is a mess otherwise)
            molecule_name = self.molecule_dictionary[self.synthesis_data["molecule_names"][index]]
           
            # get the spectral data, that matches "sample_number".txt and read in the peak position and FWHM
            path = self.spectrum_path + sample_number + ".txt"
            if not os.path.exists(path): 
                continue
            fwhm, peak_pos, spectrum, wavelength = self.read_spectrum(sample_number)

            # if the spectrum is used as input, it should be compressed to reduce the dimensionality
            spectrum = compress_spectrum(spectrum=spectrum[0:150])                          


            # check flags (would be more elegant to do this during the csv read in, but this is more flexible for now)
            if self.molecule != "all" and self.molecule != molecule_name:
                continue  
            if peak_pos <= min(self.wavelength_filter) or peak_pos >= max(self.wavelength_filter):
                continue
            if self.target == "PLQY" and self.synthesis_data['PLQY'][index] == 0:
                continue
            if self.exclude_no_star and self.synthesis_data['include PLQY'][index] != 1 :
                continue
            if self.monodispersity_only and self.synthesis_data['monodispersity'][index] != "1":
                continue


            # classify the product
            NPL_type = self.get_NPL_type(peak_pos)
            product = self.synthesis_data["NC shape"][index]   # "NR", "NPL", "NC"


            # get global attributes df for the molecule
            global_attributes_df = self.global_attributes_df.loc[self.global_attributes_df['antisolvent'] == molecule_name]
            
            
            # calculate extra attributes (relative polarity needs to be denormalized first)
            RelPolarity = global_attributes_df['relative polarity (-)'].to_numpy().astype(float)[0]
            RelPolarity = self.denormalize(RelPolarity, "relative polarity (-)")
            polarity = ((self.synthesis_data["V (antisolvent)"][index] * RelPolarity) 
                        + ( 0.099 * self.synthesis_data["V (PbBr2 prec.)"][index])) / (self.synthesis_data["V (antisolvent)"][index] + self.synthesis_data["V (PbBr2 prec.)"][index] + 0.0001)
            special_ratio = (self.synthesis_data["V (antisolvent)"][index] * RelPolarity) / (self.synthesis_data["V (PbBr2 prec.)"][index] + self.synthesis_data["c (PbBr2)"][index] + 0.0001)
            

            # select the global attributes, add the extra attributes
            global_attributes = [global_attributes_df[attribute].to_numpy().astype(float)[0] for attribute in self.global_attribute_selection]
            

            # pass the selected synthesis parameters to a list
            synthesis_parameters = []
            for key in self.synthesis_training_selection:
                synthesis_parameters.append(self.synthesis_data[key][index])


            # total parameters
            #synthesis_parameters.append(polarity)        #---------------------->  add the extra attributes here; 
                                                                                  # BUT: also add them to the global_attribute_selection !!!
            total_parameters = synthesis_parameters + global_attributes


            # encode the molecules (one_hot or geometry)
            encoding = self.encode(molecule_name, self.encoding)


            # set the target
            if self.target   ==     "FWHM": target = fwhm
            elif self.target ==     "PEAK_POS": target = peak_pos
            elif self.target ==     "NPL_TYPE": target = NPL_type
            elif self.target ==     "PLQY": target = self.synthesis_data["PLQY"][index]
            elif self.target ==     "SYNTH": target = self.synthesis_data["Pb_Cs_ratio"][index]     # for reverse approach
            elif self.target ==     "NR": target = 1 if product == "NR" else 0
            else:                   raise ValueError("No valid target specified")
            
        
            ### ----------------------  DATA OBJECT CREATION  ---------------------- ###
             # ( this is the final output format of the data, good practice to exclued everything you don't need)

            data_point = {  "y": target,
                            "spectrum": spectrum,
                            "peak_pos": peak_pos,
                        #   "NPL_type": NPL_type,
                            "fwhm": fwhm,
                            "sample_number": sample_number,
                            "total_parameters": total_parameters,
                        #   "global_attributes": global_attributes,
                            "synthesis_parameters": synthesis_parameters,
                            "parameter_selection": self.total_training_parameter_selection,
                            "encoding": encoding,
                            "molecule_name": molecule_name,
                        #   "product": product,
                            }
            
            self.data.append(data_point)
        
        # normalize the spectral data (discouraged when working with different datastructure objects, since the normalization is selection dependent)
        for data in self.data:
            data["fwhm"] = (data["fwhm"] - self.max_min["FWHM"][1]) / (self.max_min["FWHM"][0] - self.max_min["FWHM"][1])

        return self.data
    


#### ------------------------------------------  helper functions  --------------------------------------------- ####

    def encode(self, molecule_name, encoding_type):
        """ 
            encodes the molecule name to a one hot or geometry encoding 
        """

        # this is a more complex encoding based on the molecule geometry ( see molecule_geometry list)
        if encoding_type == "geometry":
            encoding = None
            for molecule in self.molecule_geometry:
                if molecule['molecule'] == molecule_name:
                    encoding =  [molecule['chainlength']] + [molecule['cycles']] + [molecule['group_pos']] + molecule['group']
            
            if encoding is None: 
                raise KeyError(f"No geometry found for {molecule_name}")


        # as the name suggests, this is just a one hot encoding based on the molecule name list
        elif encoding_type == "one_hot":   
            encoding = [0] * len(self.molecule_names)
            encoding[self.molecule_names.index(molecule_name)] = 1

        else:
            raise ValueError("No valid encoding specified")

        return encoding
    

    def read_spectrum(self, sample_number):
        """
            Reads in the spectrum of a single sample and returns the FWHM and peak position
        """
        
        #path
        path = self.spectrum_path + sample_number + ".txt"
    
        wavelength, spectrum = [], []
        with open(path, "r") as filestream:
            for line in filestream:

                if line == '': break
                x, A, y  = line.split(",")                          # of the form (wavelength in nm, amplitude, norm. amplitude)

                if self.wavelength_unit == "EV":
                    wavelength.append(nm_to_ev(float(x)))           
                else:
                    wavelength.append(float(x)) 
                spectrum.append(float(y))

        # get peak position
        max_index = spectrum.index(max(spectrum))
        peak_pos = wavelength[max_index]

        #logic for FWHM, could be changed
        half_max = 0.5 * max(spectrum)
        left_index, right_index = max_index, max_index

        while spectrum[left_index] > half_max:
            left_index -= 1
        while spectrum[right_index] > half_max:
            right_index += 1

        fwhm = abs(wavelength[left_index] - wavelength[right_index])


        # update max_min dictionary (normalization happens during the data creation)
        if fwhm > self.max_min["FWHM"][0]:          self.max_min["FWHM"][0] = fwhm
        if fwhm < self.max_min["FWHM"][1]:          self.max_min["FWHM"][1] = fwhm
        
        return fwhm, peak_pos, spectrum, wavelength



    def normalize(self, a, name):
        """
            norm. dataframes or arrays and stores the max and min values for denormalization
        """
        self.max_min[name] = [a.max(), a.min()]

        if (a.max() - a.min()) == 0: 
            print(f"Normalization error: {a}  has no range; returning original array")
            return a
        
        return (a-a.min())/(a.max()-a.min())
    

    def denormalize(self, a, name):
        """
            denormalizes a value based on the max and min values
        """
        try:
            max_val, min_val = self.max_min[name]
        except KeyError:
            print(f"KeyError: {name} not found in max_min dictionary")
        return a * (max_val - min_val) + min_val


    def get_NPL_type(self, peak_pos):
        """
            classify the NPL type from the peak position using the ml_dictionary
        """
        
        if self.wavelength_unit == "EV":
            peak_pos = ev_to_nm(peak_pos)

        for key, value in self.ml_dictionary.items():
            if value[0] <= peak_pos <= value[1]:
                return float(key)
            
        return 0



#### ------------------------------------------  init. helpers  --------------------------------------------- ####

    
    def read_global_attributes_and_nomalize(self):
        """
            reads in the global attributes of the molecules and normalizes them
        """
        
        # dataframe
        df = pd.read_csv( self.global_attributes_path, delimiter= ';', header= 0)

        
        # normalize the columns
        if self.normalization:
            for column in self.global_attribute_selection + ["relative polarity (-)"]:
                df[column] = self.normalize(df[column], column)
        return df
    
    
    
    def read_synthesis_data(self):
        """
            read in the synthesis data from dataframe and return as container (dict.)
        """

        # dataframe
        try:
            df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)                                                                 
        except FileNotFoundError:
            print(f"FileNotFoundError: {self.synthesis_file_path} not found")
            return None

        try:
            # calculate the Pb/Cs ratio
            Pb_Cs_ratio = (df["c (PbBr2)"] * df["V (PbBr2 prec.)"] ) / (df["c (Cs-OA)"]       * df["V (Cs-OA)"])	
            AS_Pb_ratio =  df["V (antisolvent)"]                     / (df["V (PbBr2 prec.)"] * df["c (PbBr2)"])
            AS_Pb_ratio /= 100    # change of units

            # normalize the columns 
            if self.normalization:
                for key in self.synthesis_selection:
                    df[key] = self.normalize(df[key], key)

            # create the synthesis data container
            synthesis_data = {key  : df[key].to_numpy().astype(float) for key in self.synthesis_selection} 

            # add ratios (probably the most expressive parameters)
            synthesis_data["Pb_Cs_ratio"] = Pb_Cs_ratio.to_numpy().astype(float)
            synthesis_data["AS_Pb_ratio"] = AS_Pb_ratio.to_numpy().astype(float)

            # add target related data
            synthesis_data["monodispersity"] = df["monodispersity"].to_numpy()
            synthesis_data["sample_numbers"] = df['Sample No.'].to_numpy()
            synthesis_data["PLQY"] =           df["PLQY"].to_numpy().astype(float)
            synthesis_data["include PLQY"] =   df["include PLQY"].to_numpy()
            synthesis_data["NC shape"] =       df["NC shape"].to_numpy()

            # add the sample numbers and molecule names (not normalized obviously)
            synthesis_data["molecule_names"] = df['antisolvent'].to_numpy()
        
        except KeyError: print("KeyError: invalid key encountered in the synthesis data")

        return synthesis_data



#### ------------------------------------------  PLOTTING  --------------------------------------------- ####

    def plot_data(self, var1, var2, var3):
        """
            Scatter plot of the data in parameter space, for visualization purposes
        """
        index1 = self.synthesis_training_selection.index(var1)
        index2 = self.synthesis_training_selection.index(var2)
        index3 = self.synthesis_training_selection.index(var3)

        x = [data["total_parameters"][index1] for data in self.data]
        y = [data["total_parameters"][index2] for data in self.data]
        z = [data["total_parameters"][index3] for data in self.data]
        c = [data["y"] for data in self.data]

        # plot with plotly
        fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=c, colorscale='Viridis', opacity=0.8)))
        fig.update_layout(scene = dict(xaxis_title=var1, yaxis_title=var2, zaxis_title=var3))
        fig.show()
        fig.write_html(f"data/NR_{var1}_{var2}_{var3}peak_pos_3D_plot.html")



    def plot_As_Pb_peak_pos(self, molecule_name):
        """
            Plots the peak position over AS/Pb ratio
        """
        
        AS_Pb =    [data["total_parameters"][0] for data in self.data if data["molecule_name"] == molecule_name]
        peak_pos = [data["peak_pos"] for data in self.data if data["molecule_name"] == molecule_name]
        
        if len(AS_Pb) == 0:
            raise ValueError(f"Not enough data found for {molecule_name}")

        # fitting a Sigmoid function
        Sigmoid = lambda x, a, b, c, d: (a / (1 + np.exp(-b * (x - c)))) + d

        if self.wavelength_unit == "EV":
              bounds = ([0.25, -0.04, 0, 2.35], [0.35, 0, 300, 2.45])
        else: bounds = ([50, 0, 0, 460], [65, 0.3, 300, 463])

        popt, pcov = curve_fit(Sigmoid, AS_Pb, peak_pos, bounds=bounds)
        print(f"Optimal parameters: {popt}")


        #plotting
        fig, ax = plt.subplots()
        x_vec = np.linspace(min(AS_Pb), max(AS_Pb), 500)
        label = f"y = $\\frac{{{round(popt[0],2)}}}{{1 + e^{{-{round(popt[1],2)}(x - {round(popt[2],2)})}}}} + {round(popt[3],2)}$"

        ax.plot(x_vec, Sigmoid(x_vec, *popt), color="red", label = label)
        ax.scatter(AS_Pb, peak_pos)

        ax.set_xlabel("AS/Pb ratio")
        ax.set_ylabel("Peak Position")
        ax.set_title(f"Peak Position vs AS/Pb ratio for {molecule_name}")
        ax.legend(fontsize = 15)

        return fig, ax



    def plot_As_Pb_peak_pos_3D(self, molecule_selection  = None):
        """
            3D plot of the As/Pb ratio vs. peak position vs. molecule property
        """

        if molecule_selection is None: molecule_selection = self.molecule_names

        AS_Pb =                 [data["total_parameters"][0]  for data in self.data if data["molecule_name"] in molecule_selection]
        peak_pos =              [data["peak_pos"]             for data in self.data if data["molecule_name"] in molecule_selection]
        color =                 [data["encoding"][-1]         for data in self.data if data["molecule_name"] in molecule_selection]

        for n in range(10):

            molecule_property =     [data["global_attributes"][n] for data in self.data if data["molecule_name"] in molecule_selection]

            #plot with plotly
            fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Scatter3d(x=AS_Pb, y=peak_pos, z=molecule_property, mode='markers', marker=dict(size=5, color=color, colorscale='Viridis', opacity=0.8)))
            fig.update_layout(scene = dict(xaxis_title='AS/Pb ratio', yaxis_title='Peak Position', zaxis_title=self.global_attribute_selection[n]))
            fig.show()
            fig.write_html(f"data/{n}_3D_plot.html")



    def plot_avg_target(self):
        """
            Plots the average target value for each molecule and NPL type as a heatmap
        """

        map = np.zeros((len(self.molecule_names), 9))
        stds = map.copy()

        for molecule in self.molecule_names:
            for NPL_type in range (1, 10):

                targets = [data["y"] for data in self.data if data["molecule_name"] == molecule and data["NPL_type"] == NPL_type]

                if len(targets) == 0:
                    continue

                stds[self.molecule_names.index(molecule), NPL_type - 1] = np.std(targets)
                map[self.molecule_names.index(molecule), NPL_type - 1] =  np.mean(targets)
        

        map = map[~np.all(map == 0, axis=1)]
        fig, ax = plt.subplots()

        im = ax.imshow(map, cmap="viridis")
        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_title(f"Average {self.target} Distribution")
        ax.set_xticks(np.arange(9))
        ax.set_yticks(np.arange(len(self.molecule_names)))
        ax.set_xticklabels([str(i) for i in range(1, 10)])
        ax.set_yticklabels(self.molecule_names)

        plt.show()



### ------------------------DICIONARIES------------------------ ###


    def get_molecule_dictionary(self):
        """
            dictionaries for the molecule names, atom to number and number to atom conversion
        """

        ml_dictionary = {"1": (402, 407),
                         "1.5": (408, 429),
                         "2": (430, 437),
                         "2.5": (438, 456),
                         "3": (457, 466),
                         "3.5": (467, 471),
                         "4": (472, 481),
                         "4.5": (482, 483),
                         "5": (484, 489),
                         "6": (491, 497),
                         "7": (498, 504),
                         "8": (505, 509),
                         "9": (510, 525),}

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
        

        molecule_geometry = [
                             {'molecule': "Methanol",       'group': [1,0], 'chainlength': 0.1, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 2.87},
                             {'molecule': "Ethanol",        'group': [1,0], 'chainlength': 0.2, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 1.15},
                             {'molecule': "Propanol",       'group': [1,0], 'chainlength': 0.3, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 0.75},
                             {'molecule': "Isopropanol",    'group': [1,0], 'chainlength': 0.3, 'cycles' : 0, 'group_pos': 1, 'diffusivity' : 0.56},
                             {'molecule': "Butanol",        'group': [1,0], 'chainlength': 0.4, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 0.47},
                             {'molecule': "Acetone",        'group': [0,1], 'chainlength': 0.3, 'cycles' : 0, 'group_pos': 1},
                             {'molecule': "Butanone",       'group': [0,1], 'chainlength': 0.4, 'cycles' : 0, 'group_pos': 1},
                             {'molecule': "Cyclopentanone", 'group': [0,1], 'chainlength': 0.5, 'cycles' : 1, 'group_pos': 0},
        ]


        
        atom_to_num = {'H' : 1,
              'C' : 6,
              'O' : 8,
              'N' : 7,
              'S' : 16
              }
        
        num_to_atom = {v: k for k, v in atom_to_num.items()}   #inverse dictionary

        return molecule_dictionary, atom_to_num, num_to_atom, ml_dictionary, molecule_geometry
