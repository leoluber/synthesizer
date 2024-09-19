r"""  Datastructure.py implements the Datastructure class used 
      throughout the project as main data structure  """
      # << github.com/leoluber >> 


from typing import Literal
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly

# custom
from helpers import *



class Datastructure:

    """ Datastructure class

    General purpose data structure that reads in the synthesis data, spectral information 
    and the global attributes of the molecules, performs various selections and creates a 
    list of dict. objects that can be used for machine learning tasks.
    Can be easily extended to include more attributes or other target values.
    It is discouraged to use this class with different data sets!

    REQUIREMENTS
    ------------
    - all data should be at "/data/" relative to the current working directory and contain:
        - a folder "spectrum" with the spectral data in .txt files (csv)
        - AntisolventProperties.csv with the global attributes of the molecules; names for 
          properties to be found in the global_attribute_selection list
        - a _.csv file with the synthesis data (synthesis_file_path) containing the sample 
          numbers, molecule names, synthesis parameters and other properties

    PARAMETERS
    ----------
    - synthesis_file_path:   path to the synthesis data file (csv)
    - target:                target value for the machine learning task (FWHM, PEAK_POS, PLQY, NR)
    - encoding:              encoding type for the molecules (one_hot, geometry)
    - wavelength_unit:       unit of the wavelength (NM, EV)
    - wavelength_filter:     filter for the peak position (list of two values in nm)
    - molecule:              molecule to be selected (all, specific molecule)
    - add_baseline:          add a baseline data to the data objects
    - PLQY_criteria:         exclude samples with low Cs/Pb ratios if the measurement can not be trusted
    - monodispersity_only:   exclude samples that are not monodisperse
    - P_only:                exclude samples that are not P type

    INTENDED USAGE
    -------------
    >>> from Datastructure import Datastructure
    >>> ds = Datastructure(synthesis_file_path = "synthesis_data.csv", target = "FWHM")
    >>> data_objects = ds.get_data()

    DATA OBJECTS
    ------------
    Dictionary objects with the following keys (returned by get_data())

    - "y":                   target value
    - "total_parameters":    list of all selected parameters (list)
    - "parameter_selection": list of selected parameters as strings (for reference)
    - "encoding":            one hot OR geometric encoding of the molecule (list)
    - "sample_number":       sample number (string)
    - "molecule_name":       name of the molecule (string)
    - "monodispersity":      monodispersity (bool)
    - "product":             product type
    - "index":               index of the data object
    - "age_prec":            age of the precursor

    """


    def __init__(
                 self,
                 synthesis_file_path:   str, 

                 target:          Literal["FWHM", "PEAK_POS", "PLQY", "NR"] = "FWHM",
                 encoding:        Literal["one_hot", "geometry"] = "one_hot",
                 wavelength_unit: Literal["NM", "EV"] = "NM",

                 wavelength_filter =    [400, 600],
                 molecule =            "all",
                 add_baseline =         False,
                 PLQY_criteria =        True,
                 monodispersity_only =  False,
                 P_only =               False,

                 ):
        

        # main stettings
        self.target          = target
        self.encoding        = encoding
        self.add_baseline    = add_baseline
        self.wavelength_unit = wavelength_unit


        # selection flags
        self.flags = {"PLQY_criteria"      : PLQY_criteria,
                      "monodispersity_only": monodispersity_only,
                      "P_only"             : P_only,
                      "molecule"           : molecule}
        

        # directories
        self.current_path =             os.getcwd()
        self.data_path =                self.current_path + "/data/"
        self.synthesis_file_path =      self.data_path + synthesis_file_path
        self.global_attributes_path =   self.data_path + "AntisolventProperties.csv"
        self.spectrum_path =            self.data_path + "spectrum/"


        # dictionaries for molecule names, geometries and atom numbers
        self.molecule_dictionary, self.ml_dictionary = self.get_molecule_dictionary()
        self.molecule_geometry = self.get_molecule_geometry()


        # list of all relevant molecule names
        self.molecule_names = list(self.molecule_dictionary.values())


        # data related local variables (should not be accessed directly)
        self.data = []    
        self.max_min = {}  # norm/denorm
        if wavelength_unit == "EV":
            self.wavelength_filter = [nm_to_ev(wavelength_filter[1]),
                                      nm_to_ev(wavelength_filter[0])]
        else:
            self.wavelength_filter = wavelength_filter

        # ------- Training parameter selection ------- #

        """ Can be adjusted after initialization BUT before calling get_data()! """

        self.global_attribute_selection =       [] # "dielectric constant (-)", 'relative polarity (-)',"Hansen parameter hydrogen bonding (MPa)1/2",]
        self.synthesis_training_selection =     ["c (PbBr2)", "V (antisolvent)", "V (Cs-OA)", "V (PbBr2 prec.)", "AS_Pb_ratio",  "Cs_Pb_ratio",]
        self.total_training_parameter_selection = self.synthesis_training_selection + self.global_attribute_selection

        # ------------------------------------------- #


        # read in the data
        self.synthesis_data = self.read_synthesis_data()
        self.global_attributes_df = self.read_global_attributes_and_nomalize()


        # densities for adjusting the concentrations
        self.densities = {f"{molecule}": self.global_attributes_df
                          .loc[self.global_attributes_df['antisolvent'] == molecule]['n [mol/L]']
                          .to_numpy()
                          .astype(float)[0] for molecule in self.molecule_names}




    def get_data(self)-> list:
        
        """ Data collection function
        
        Returns a list of data objects (dicts) that can be used for machine learning tasks
        - Loop that iterates through the samples in the synthesis file, reads in all the extra information and 
            represents it as a Data object
        - synthesis data and the gloal attributes dataframe are read in during initialization 
        (accessed via self.synthesis_data["property_name"][index])
        - molecule specific data is read in during the loop

        """


        for index, sample_number in enumerate(self.synthesis_data["sample_numbers"]):


            # check if the molecule is in the molecule dictionary, skip if not
            if self.synthesis_data["molecule_names"][index] not in self.molecule_dictionary.keys():
                continue

            # translates the data dependent molecule name to inherent module specific names 
            molecule_name = self.molecule_dictionary[self.synthesis_data["molecule_names"][index]]


            # get the spectral data that matches the sample_number (if it exists)
            path = self.spectrum_path + sample_number + ".txt"
            if not os.path.exists(path): 
                continue
            fwhm, peak_pos, spectrum, wavelength = self.read_spectrum(path)            


            # BASELINE->  if baseline is set to True, the Toluene samples are included
            # the "molecule" == "all" case is handled outside the loop
            if molecule_name == "Toluene" and self.add_baseline and self.flags["molecule"] != "all":
                molecule_name = self.flags["molecule"]


            # check flags (more elegant to do this during the csv reading, but this gives allows
            # resetting the flags and calling get_data() again for the same Datastructure object
            if not self.check_flags(index, molecule_name, peak_pos):
                continue


            """
                PLQY data is an unfortunate mess, the following is a temporary solution 
                --> excluding samples with low Cs/Pb ratios if the measurement can not be trusted 
                (old prec, wrong timing)
            """
            if self.flags["PLQY_criteria"]:
                if (self.synthesis_data["Cs_Pb_ratio"][index] < 0.2 
                    and self.synthesis_data["Age of prec"][index] not in ["0",] 
                    and self.target == "PLQY"):
                    continue


            # classify the product
            NPL_type = self.get_NPL_type(peak_pos)             # Thickness of the NPLs in MLs
            product = self.synthesis_data["NC shape"][index]   # "NR", "NPL", "NC"


            # get global attributes df for the molecule
            global_attributes_df = self.global_attributes_df.loc[self.global_attributes_df['antisolvent'] == molecule_name]


            # adjust th As/Pb ratio for actual amount of substance
            self.synthesis_data['AS_Pb_ratio'][index] = (self.synthesis_data['AS_Pb_ratio'][index] 
                                                        * self.densities[molecule_name] / 10000)
            self.synthesis_data['Cs_As_ratio'][index] = (self.synthesis_data['Cs_As_ratio'][index] 
                                                         * self.densities[molecule_name]) 


            # read selected attributes to a list format
            global_attributes = [global_attributes_df[attribute]
                                 .to_numpy()
                                 .astype(float)[0] for attribute in self.global_attribute_selection]
            
            synthesis_parameters = [self.synthesis_data[key][index] 
                                    for key in self.synthesis_training_selection]

            total_parameters = synthesis_parameters + global_attributes


            # encode the molecules (one_hot or geometry)
            encoding = self.encode(molecule_name, self.encoding)
            if encoding is None: continue


            # set the target
            match self.target:
                case "NPL_TYPE": target = NPL_type
                case "FWHM": target = fwhm
                case "PEAK_POS": target = peak_pos
                case "PLQY": target = self.synthesis_data["PLQY"][index]
                case "NR": target = 1 if product == "NR" else 0
                case _: raise ValueError("No valid target specified")
            
            
        
            ### ----------------------  DATA OBJECT CREATION  ---------------------- ###
            # ( this is the final output format of the data,
            # good practice to exclued everything you don't need)

            data_point = {  "y": target,
                            "spectrum": (wavelength, spectrum),
                            "peak_pos": peak_pos,
                            "Cs_Pb_ratio": self.synthesis_data["Cs_Pb_ratio"][index],
                            "suggestion":  self.synthesis_data["suggestion"][index],
                            "S/P":         self.synthesis_data["S/P"][index],
                            "fwhm": fwhm,
                            "sample_number": sample_number,
                            "plqy": self.synthesis_data["PLQY"][index],
                            "total_parameters": total_parameters,
                            "encoding": encoding,
                            "molecule_name": molecule_name,
                            "monodispersity": self.synthesis_data["monodispersity"][index],
                            "index" : index,
                            "Cs_Pb_ratio": self.synthesis_data["Cs_Pb_ratio"][index],
                            }
            
            self.data.append(data_point)

        # add baseline
        if self.add_baseline:
            self.data = self.add_limit_baseline(self.data)
            self.data = self.add_Toluene_baseline(self.data)
        
        return self.data 
    


#### ------------------------------------  HELPERS  ---------------------------------- ####


    def encode(self, molecule_name, encoding_type) -> list:

        """ Encodes the molecule name to a one hot or geometry encoding """


        # (1) this is a more complex encoding based on the molecule geometry ( see molecule_geometry list)
        if encoding_type == "geometry":
            encoding = None
            for molecule in self.molecule_geometry:
                if molecule['molecule'] == molecule_name:
                    encoding =  ([molecule['chainlength']] + molecule['group'] 
                                 + [molecule['group_pos']] + [molecule['cycles']])
            
            if encoding is None: 
                raise ValueError(f"Geometry encoding not found for {molecule_name}")


        # (2) as the name suggests, this is just a one hot encoding based on the molecule name list
        elif encoding_type == "one_hot":   
            encoding = [0] * len(self.molecule_names)
            encoding[self.molecule_names.index(molecule_name)] = 1

        else:
            raise ValueError("No valid encoding specified")

        return encoding
    


    def read_spectrum(self, path) -> tuple:

        """ Reads in the spectral data from a .txt file
        
        Critically important to have the data in the correct format, that is: 
        - wavelength in the first column
        - spectrum in the third column!
        - comma separated
        """
    
        wavelength, spectrum = [], []
        with open(path, "r") as filestream:
            for line in filestream:

                if line == '': break
                x, A, y  = line.split(",")

                if self.wavelength_unit == "EV":
                    wavelength.append(nm_to_ev(float(x)))           
                else:
                    wavelength.append(float(x)) 
                spectrum.append(float(y))


        """ TODO: both of the following operations could be done more elegantly
            both should ideally be evaluated by fitting a function, but since the
            data has very little noise, this is fine
        """ 

        # get peak position
        max_index = spectrum.index(max(spectrum))
        peak_pos  = wavelength[max_index]

        #logic for FWHM, could be changed
        half_max  = 0.5 * max(spectrum)
        left_index, right_index = max_index, max_index

        while spectrum[left_index] > half_max:
            left_index -= 1
        while spectrum[right_index] > half_max:
            right_index += 1

        fwhm = abs(wavelength[left_index] - wavelength[right_index])
        
        return fwhm, peak_pos, spectrum, wavelength



    def normalize(self, a, name):

        """ norm. dataframes or arrays and stores the max and min values for denormalization """

        self.max_min[name] = [a.max(), a.min()]

        if (a.max() - a.min()) == 0: 
            print(f"Normalization issue: {a}  has no range; returning original array")
            return a
        
        return (a-a.min())/(a.max()-a.min())
    


    def denormalize(self, a, name):

        """ denormalizes a value based on the max and min values """

        try:
            max_val, min_val = self.max_min[name]
        except KeyError:
            print(f"KeyError: {name} not found in max_min dictionary")

        return a * (max_val - min_val) + min_val



    def get_NPL_type(self, peak_pos) -> float:

        """ Classify the NPL type from the peak position using the ml_dictionary """
        
        if self.wavelength_unit == "EV":
            peak_pos = ev_to_nm(peak_pos)

        for key, value in self.ml_dictionary.items():
            if value[0] <= peak_pos <= value[1]:
                return float(key)
            
        return 0.



#### ------------------------------  init. helpers  ------------------------------- ####

    
    def read_global_attributes_and_nomalize(self) -> pd.DataFrame:

        """ Reads in the global attributes of the molecules and normalizes them """
        

        # dataframe
        df = pd.read_csv( self.global_attributes_path, delimiter= ';', header= 0)
        
        # normalize the columns
        for column in self.global_attribute_selection + ["relative polarity (-)"]:
            df[column] = self.normalize(df[column], column)
        return df
    
    

    def read_synthesis_data(self) -> dict:

        """ Read in the synthesis data from dataframe and return as container (dict.) 
        
        The synthesis data is read in from the synthesis file and normalized.
        The data is then stored in a dictionary container:
        - the keys are the column names of the synthesis data file
        - the values are the normalized data as numpy arrays
        --> ratios are NOT normalized, use units instead to rescale them
        """

        synthesis_selection = [ "c (PbBr2)", "c (OlAm)", "c (OA)", "V (Cs-OA)", 
                               "c (Cs-OA)" ,"V (antisolvent)", "V (PbBr2 prec.)"]  


        # dataframe
        try:
            df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)  

        except FileNotFoundError:
            print(f"FileNotFoundError: {self.synthesis_file_path} not found")
            return None
        
        # remove unwanted molecules
        recognized_molecules = list(self.molecule_dictionary.keys())
        df = df[df['antisolvent'].isin(recognized_molecules)]


        try:
            # calculate the ratios 
            Pb_Cs_ratio = ((df["c (PbBr2)"] * df["V (PbBr2 prec.)"] ) 
                           / (df["c (Cs-OA)"] * df["V (Cs-OA)"]))
            Cs_Pb_ratio = 1 / Pb_Cs_ratio
            AS_Pb_ratio = (df["V (antisolvent)"]) / ((df["V (PbBr2 prec.)"] * df["c (PbBr2)"]))
            Cs_As_ratio = (df["c (Cs-OA)"] * df["V (Cs-OA)"]) / (df["V (antisolvent)"] + 0.0001)

            # normalize the columns
            for key in synthesis_selection:
                df[key] = self.normalize(df[key], key)

            # create the synthesis data container
            synthesis_data = {key  : df[key].to_numpy().astype(float) for key in synthesis_selection} 

            # add ratios (probably the most expressive parameters)
            synthesis_data["AS_Pb_ratio"] = AS_Pb_ratio.to_numpy().astype(float)
            synthesis_data["Cs_Pb_ratio"] = Cs_Pb_ratio.to_numpy().astype(float)
            synthesis_data["Cs_As_ratio"] = Cs_As_ratio.to_numpy().astype(float)

            # add target related data
            synthesis_data["monodispersity"] = df["monodispersity"].to_numpy().astype(int)
            synthesis_data["sample_numbers"] = df['Sample No.'].to_numpy()
            synthesis_data["PLQY"]           = df["PLQY"].to_numpy().astype(float)
            synthesis_data["include PLQY"]   = df["include PLQY"].to_numpy()
            synthesis_data["NC shape"]       = df["NC shape"].to_numpy()
            synthesis_data["S/P"]            = df["S/P"].to_numpy()
            synthesis_data["suggestion"]     = df["suggestion"].to_numpy().astype(str)
            synthesis_data["suggestion"]     = ["" if x == "nan" else x for x in synthesis_data["suggestion"]]
            synthesis_data["Age of prec"]    = df["Age of prec"].to_numpy()

            # add the sample numbers and molecule names (not normalized obviously)
            synthesis_data["molecule_names"] = df['antisolvent'].to_numpy()
        

        except KeyError:
            print(f"maybe a key error, maybe not a key error, but life's too short for proper error propagation")
            return None

        return synthesis_data



    def check_flags(self, index, molecule_name, peak_pos) -> bool:

        """ check the flags for the data selection """


        if self.flags["molecule"] != "all" and self.flags["molecule"] != molecule_name:
            return False
        
        if peak_pos <= min(self.wavelength_filter) or peak_pos >= max(self.wavelength_filter):
            return False
        
        if self.target == "PLQY" and self.synthesis_data['PLQY'][index] == 0:
            return False
        
        if self.target == "PLQY"  and  self.synthesis_data['include PLQY'][index] != 1 :
            return False
        
        if self.flags["monodispersity_only"] and self.synthesis_data['monodispersity'][index] != 1:
            return False
        
        if self.flags["P_only"] and self.synthesis_data['S/P'][index] != "P": 
            return False
        
        if self.synthesis_data["Cs_Pb_ratio"][index] > 1:
            return False
        
        return True



    def add_limit_baseline(self, data_objects):

        """ Adds a baseline to the data objects """
    

        # get the 515nm baseline
        inputs  = [[i/10, 1] for i in range(0, 8)]
        inputs += [[0.8, i/10] for i in range(4, 10)]
        
        peak  = 510
        if self.wavelength_unit == "EV":
            peaks = [nm_to_ev(peak) for i in range(len(inputs))]
        else:
            peaks = [peak for i in range(len(inputs))]

        # add new data to data objects
        molecules = list(set([data["molecule_name"] for data in data_objects]))

        for molecule in molecules:
            for input, peak in zip(inputs, peaks):
                data_objects.append({"molecule_name": molecule, "y": peak, 
                                     "encoding": self.encode(molecule, self.encoding),
                                    "total_parameters": input, "spectra": None, 
                                    "peak_pos": peak, "index": -1, 
                                    "sample_number": "baseline"})

        return data_objects



    def add_Toluene_baseline(self, data_objects):

        """ Adds a baseline of the Toluene (no antisolvent) data to the data objects """


        # get Toluene data
        toluene_data = [data for data in data_objects if data["molecule_name"] == "Toluene"]

        # define the molecule selection
        if self.flags["molecule"] == "all":
            molecule_selection = self.molecule_names.copy()
            molecule_selection.remove("Toluene")
        else:
            molecule_selection = [self.flags["molecule"]]

        # add the Toluene data to the data objects
        for molecule in molecule_selection:
            for data in toluene_data:
                data["molecule_name"] = molecule
                data["encoding"] = self.encode(molecule, self.encoding)
                data_objects.append(data)

        return data_objects



#### ---------------------------  PLOTTING  ------------------------------ ####
    """       < don't put this on the final .git, it's a mess >         """


    def plot_data(self, var1, var2, var3 = None, 
                  kernel = None, 
                  model = "KRR", 
                  molecule = "all") -> None:

        """
            Scatter plot of the data in parameter space, for visualization;
            kernel can be plotted as well
            -> var1, var2, var3: parameters to be plotted
        """

        # get the indices of the parameters
        index1 = self.synthesis_training_selection.index(var1)
        index2 = self.synthesis_training_selection.index(var2)

        # get the data
        if molecule == "all":
            x =     [data["total_parameters"][index1] for data in self.data]
            y =     [data["total_parameters"][index2] for data in self.data]
            t =     [data["y"] for data in self.data]
            molecules = [self.molecule_names.index(data["molecule_name"]) for data in self.data]

        else:
            x = [data["total_parameters"][index1] for data in self.data 
                 if data["molecule_name"] == molecule]
            y = [data["total_parameters"][index2] for data in self.data 
                 if data["molecule_name"] == molecule]
            t = [data["y"] for data in self.data 
                 if data["molecule_name"] == molecule]
            molecules = [1 for _ in self.data]


        # plot data with plotly
        fig = plotly.graph_objects.Figure()
        fig.add_trace(plotly.graph_objects
                      .Scatter3d(x=x, y=y, z=t, mode='markers', 
                                marker=dict(size=16, color=molecules, 
                                            colorscale='Viridis', opacity=0.8)))
        fig.update_layout(scene = dict(xaxis_title=var1, 
                                       yaxis_title=var2, 
                                       zaxis_title=var3))

        # plot model
        if kernel is not None:
            y_vec = np.linspace(0, 1, 100)
            x_vec = np.linspace(0, 0.8, 100)
            X, Y  = np.meshgrid(x_vec, y_vec)
            input = np.c_[X.ravel(), Y.ravel()]
            
            # add self.encode(molecule, self.encoding) to the input
            if molecule != "all":
                encoding = self.encode(molecule, self.encoding)
                input = [np.append(encoding, row ) for row in input]
            else:
                fig.show()
                return fig

            if model == "GP":
                input = np.array(input)
                print(input.shape)
                Z = kernel.model.predict(input)[0].reshape(X.shape)
                err = kernel.model.predict(input)[1].reshape(X.shape)

            elif model == "KRR":
                Z = kernel.predict(input).reshape(X.shape)

            # add the surface plot of the kernel
            fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=0.5, colorscale='gray'))
            
            # add confidence intervals
            #if model == "GP":
                #fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z + 1.96 * err, opacity=0.5, colorscale='gray'))
                #fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z - 1.96 * err, opacity=0.5, colorscale='gray'))


        # save the plot as interactive html
        #molecules = list(set(molecules))
        #molecule_str = "_".join([self.molecule_names[mol] for mol in molecules])
        #fig.write_html(f"data/{var1}_{var2}_{var3}_{molecule_str}.html")


        fig.show()
        return fig
        


    def plot_2D_contour(self, var1, var2, kernel = None) -> None:

        """
            2D contour plot of the data in parameter space, for visualization 
            purposes color coded by the target value (PEAK_POS)
        """

        # get the data
        x = [data["total_parameters"][-1] for data in self.data]
        y = [data["total_parameters"][-2] for data in self.data]

        # output string
        molecules = list(set([data["molecule_name"] for data in self.data]))
        mol_str = "_".join(molecules)


        # a contour plot of the kernel
        fig, ax = plt.subplots()
        y_vec = np.linspace(0, 1, 100)
        x_vec = np.linspace(0, 8, 100)
        X, Y = np.meshgrid(x_vec, y_vec)

        # evaluate the kernel on the grid
        if kernel is not None:
            Z = kernel.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
            c = ax.contourf(X, Y, Z, 20, cmap='viridis', vmin = 440, vmax = 500)
            
            fig.colorbar(c, ax=ax, label = "PEAK POS")


        # plot the data
        ax.scatter(x, y, c = [data["y"] for data in self.data], 
                   cmap = "viridis", vmin = 440, vmax = 500, 
                   edgecolors='black')
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)

        plt.legend()
        plt.show()



    def plot_parameters(self, data_objects,) -> None:

        """
            Plots arbitrary parameters for visualization purposes
            -> not pretty, but useful for quick checks
        """


        total_1 =     	    [data["total_parameters"][0] for data in data_objects]
        total_2 =           [data["total_parameters"][1] for data in data_objects]
        peak_pos =          [data["peak_pos"]            for data in data_objects]
        target =            [data["y"]                   for data in data_objects]
        #target =           [data["y_res"]               for data in data_objects]

        molecule_name =     [self.molecule_names.index(data["molecule_name"]) 
                             for data in data_objects]
        

        if len(total_1) == 0:
            raise ValueError(f"Not enough data found for molecule: {molecule_name}")

        
        """ basic scatter plot """
        fig, ax = plt.subplots()
        cbar = plt.colorbar(ax.scatter(peak_pos, target, 
                                       c = molecule_name, cmap = "viridis_r",))
      
        """plot surface proportions"""
        # x_vec = np.linspace(min(peak_pos), max(peak_pos), 100)
        # if self.wavelength_unit == "EV":
        #     y = [surface_proportion(ev_to_nm(x)) for x in x_vec]
        # else:
        #     y = [surface_proportion(x) for x in x_vec]
        # plt.rc('legend', fontsize=2)
        # ax.plot(x_vec, y, color = "red", label = "1 - Surface Proportion", 
        #         linestyle = "dashed", linewidth = 1)

    
        ax.legend(fontsize = 15)
        plt.show()
        #plt.savefig(f"data/{molecule_name}_As_Pb_peak_pos.png")

        return fig, ax



    def plot_correlation(self) -> None:

        """
            Plots the correlation matrix of the data 
            together with the target values
        """


        data = [data["total_parameters"] for data in self.data]

        df = pd.DataFrame(data, columns = self.total_training_parameter_selection)
        df["plqy"]     = [data["plqy"] for data in self.data]
        df["fwhm"]     = [data["fwhm"] for data in self.data]
        df["peak_pos"] = [data["peak_pos"] for data in self.data]

        corr = df.corr()
        fig, ax = plt.subplots()
        im = ax.imshow(abs(corr), cmap="Blues")

        ax.set_xticks(np.arange(len(self.total_training_parameter_selection)+ 3))
        ax.set_yticks(np.arange(len(self.total_training_parameter_selection)+ 3))
        ax.set_xticklabels(self.total_training_parameter_selection 
                           + ["plqy", "fwhm", "peak_pos"], rotation='vertical')    
        ax.set_yticklabels(self.total_training_parameter_selection 
                           + ["plqy", "fwhm", "peak_pos"])


        plt.colorbar(im)
        plt.show()



#### --------------------------  DICIONARIES  ---------------------------- ####


    def get_molecule_dictionary(self) -> dict:
        
        """ Dictionaries for the molecule names and the NPL types """

        ml_dictionary = {"1": (402, 407),
                         "2": (430, 437),
                         "3": (456, 463),
                         "4": (472, 481),
                         "5": (484, 489),
                         "6": (491, 497),
                         "7": (498, 504),
                         "8": (505, 509),
                         "9": (510, 525),}
        

        molecule_dictionary = {"Tol" : 'Toluene',
                       "Ac" : "Acetone",
                       "EtOH" : "Ethanol",
                       "MeOH" : "Methanol",
                       "i-PrOH" : "Isopropanol",
                       "n-BuOH" : "Butanol",
                       "n-PrOH": "Propanol",
                       "butanone" : "Butanone",
                       "CyPen" : "Cyclopentanone",
                       "CyPol" : "Cyclopentanol",
                       "HexOH" : "Hexanol",
                       "OctOH" : "Octanol",
                       }
        
        return molecule_dictionary, ml_dictionary



    def get_molecule_geometry(self) -> list:

        """
            Geometry of the molecules (chainlength, group, cycles, group_pos) 
            TODO: move to a csv file (or just use fingerprints)
        """
        
        molecule_geometry = [
            {'molecule': "Methanol",       'group': [1,0], 'chainlength': 0.1, 'cycles' : 0, 'group_pos': 0},
            {'molecule': "Ethanol",        'group': [1,0], 'chainlength': 0.2, 'cycles' : 0, 'group_pos': 0},
            {'molecule': "Propanol",       'group': [1,0], 'chainlength': 0.3, 'cycles' : 0, 'group_pos': 0},
            {'molecule': "Isopropanol",    'group': [1,0], 'chainlength': 0.3, 'cycles' : 0, 'group_pos': 1},
            {'molecule': "Butanol",        'group': [1,0], 'chainlength': 0.4, 'cycles' : 0, 'group_pos': 0},
            {'molecule': "Acetone",        'group': [0,1], 'chainlength': 0.3, 'cycles' : 0, 'group_pos': 1},
            {'molecule': "Butanone",       'group': [0,1], 'chainlength': 0.4, 'cycles' : 0, 'group_pos': 1},
            {'molecule': "Cyclopentanone", 'group': [0,1], 'chainlength': 0.5, 'cycles' : 1, 'group_pos': 0},
            {'molecule': "Cyclopentanol",  'group': [1,0], 'chainlength': 0.5, 'cycles' : 1, 'group_pos': 0},
            {'molecule': "Hexanol",        'group': [1,0], 'chainlength': 0.6, 'cycles' : 1, 'group_pos': 0},
            {'molecule': "Octanol",        'group': [1,0], 'chainlength': 0.8, 'cycles' : 1, 'group_pos': 0},
            {'molecule': "Toluene",        'group': [0,0], 'chainlength': 0.7, 'cycles' : 1, 'group_pos': 0},
        ]

        return molecule_geometry