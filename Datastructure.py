r""" Datastructure.py implements the Datastructure class used 
     throughout the project as main data structure  """
     # << github.com/leoluber >> 




from typing import Literal
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff
import pickle
import json


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
    - PLQY_criteria:         exclude samples with low Cs/Pb ratios if they can not be trusted
    - monodispersity_only:   exclude samples that are not monodisperse
    - P_only:                exclude samples that are not P type

    BASIC USAGE
    -------------
    >>> from Datastructure import Datastructure
    >>> ds = Datastructure(synthesis_file_path = "synthesis_data.csv", target = "FWHM", ...)
    >>> ds.synthesis_training_selection = ["AS_Pb_ratio",  "Cs_Pb_ratio", ...]
    >>> data_objects = ds.get_data()

    ENCOURAGED USAGE
    ----------------
    >>> from Datastructure import Datastructure
    >>> ds = Datastructure(synthesis_file_path = "synthesis_data.csv", target = "FWHM", ...)
    >>> ds.synthesis_training_selection = ["AS_Pb_ratio",  "Cs_Pb_ratio", ...]
    >>> data_objects = ds.get_data()
    >>> ds.save_data_as_file(data_objects, "data_objects")

    >>> data_objects = ds.load_data_from_file("data_objects")


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

        target:          Literal["FWHM", "PEAK_POS", "PLQY", "MONO",] = "PEAK_POS",
        encoding:        Literal["one_hot", "geometry"] = "one_hot",
        wavelength_unit: Literal["NM", "EV"] = "NM",

        wavelength_filter =    [400, 600],
        molecule =            "all",
        add_baseline =         False,
        PLQY_criteria =        False,
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
        self.molecule_geometry = self.get_molecule_geometry()

        # ------- Training parameter selection ------- #

        self.total_training_parameter_selection  =   ["c (PbBr2)", "V (antisolvent)", 
                                                      "V (Cs-OA)", "V (PbBr2 prec.)", 
                                                      "AS_Pb_ratio",  "Cs_Pb_ratio",]
    


        # read in the data
        self.synthesis_data = self.read_synthesis_data()
        self.global_attributes_df = self.read_global_attributes_and_normalize()


        # densities for adjusting the concentrations
        self.densities = {f"{molecule}": self.global_attributes_df
                          .loc[self.global_attributes_df['antisolvent'] == molecule]['n [mol/L]']
                          .to_numpy()
                          .astype(float)[0] for molecule in self.molecule_names}



    def get_data(self)-> list:
        
        """ Data collection function
        
        Returns a list of data objects (dicts) that can be used for machine learning tasks
        - Loop that iterates through the samples in the synthesis file, reads in all the 
          extra information and represents it as a Data object
        - synthesis data and the gloal attributes dataframe are read in during initialization 
          (accessed via self.synthesis_data["property_name"][index])
        - molecule specific data is read in during the loop
        """

        for index, sample_number in enumerate(self.synthesis_data["sample_numbers"]):



            FILTER = False
            ################## TEMPORARY FIX ##################
            # FILTER = True
            # if self.synthesis_data["Age of prec"][index] != "-":
            #     continue

            if sample_number in ["j103", "j104", "j105", "j109"]:
                continue

            ###################################################
            if not FILTER:
                print("FILTER is NOT active!!!!!!!")


            # initialize the baseline label, set to False if the data is of type "baseline"
            baseline = False


            # check if the molecule is in the molecule dictionary, skip if not
            if self.synthesis_data["molecule_names"][index] not in self.molecule_dictionary.keys():
                continue


            # translates the data dependent molecule name to inherent module specific names 
            molecule_name = self.molecule_dictionary[self.synthesis_data["molecule_names"][index]]


            # get the spectral data that matches the sample_number (if it exists)
            path = self.spectrum_path + str(sample_number) + ".txt"
            if not os.path.exists(path): 
                continue
            fwhm, peak_pos, spectrum, wavelength, hm_ = self.read_spectrum(path, monodispersity = self.synthesis_data["monodispersity"][index])    


            # BASELINE->  if baseline is set to True, the Toluene samples are always included
            # the "molecule" == "all" case is handled outside the loop
            if molecule_name == "Toluene" and self.add_baseline and self.flags["molecule"] != "all":
                molecule_name = self.flags["molecule"]
                baseline = True

            
            # check flags (more elegant to do this during the csv reading, but this allows
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
            c_Perovskite = ((self.synthesis_data['c (Cs-OA)'][index]  * self.synthesis_data['V (Cs-OA)'][index] )/
                            ((self.synthesis_data['V (antisolvent)'][index]) + #self.densities[molecule_name] * 
                            ( self.synthesis_data['V (Cs-OA)'][index]  )+ #self.synthesis_data['c (Cs-OA)'][index]  *
                            ( self.synthesis_data['V (PbBr2 prec.)'][index] ))) #self.synthesis_data['c (PbBr2)'][index]  *
            self.synthesis_data['n_As'][index] = (self.synthesis_data['V (antisolvent)'][index]
                                                  * self.densities[molecule_name])
                                                  


            # read selected attributes to a list format
 
            synthesis_parameters = [self.synthesis_data[key][index] 
                                    for key in self.synthesis_training_selection]

            total_parameters = synthesis_parameters


            # encode the molecules (one_hot or geometry)
            encoding = self.encode(molecule_name, self.encoding)
            if encoding is None: continue


            # set the target
            match self.target:
                case "HM": target = hm_
                case "NPL_TYPE": target = NPL_type
                case "FWHM": target = fwhm
                case "PEAK_POS": target = peak_pos
                case "PLQY": target = self.synthesis_data["PLQY"][index]
                case "NR": target = 1 if product == "NR" else 0
                case "MONO": target = self.synthesis_data["monodispersity"][index].astype(float)
                case _: raise ValueError("No valid target specified")
            
            
            # enhanced
            enhanced = (self.synthesis_data["Age of prec"][index] not in ["-"])
        
        
            ### ----------------------  DATA OBJECT CREATION  ---------------------- ###
            # ( this is the final output format of the data,
            # good practice to exclude everything you don't need)

            data_point = {  "baseline":    baseline,
                            "artificial":  False,
                            "y": target,
                            "spectrum": (wavelength, spectrum),
                            "peak_pos": peak_pos,
                            "AS_Pb_ratio": self.synthesis_data["AS_Pb_ratio"][index],
                            "Cs_Pb_ratio": self.synthesis_data["Cs_Pb_ratio"][index],
                            "Cs_As_ratio": self.synthesis_data["Cs_As_ratio"][index],
                            "enhanced": enhanced,
                            "c_Perovskite": c_Perovskite,
                            "amount_substance": {"Cs": self.synthesis_data["n_Cs"][index],
                                                 "Pb": self.synthesis_data["n_Pb"][index],
                                                 "As": self.synthesis_data["n_As"][index]},
                            "suggestion":  self.synthesis_data["suggestion"][index],
                            "S/P":         self.synthesis_data["S/P"][index],
                            "fwhm": fwhm,
                            "sample_number": sample_number,
                            "index": index,
                            "plqy": self.synthesis_data["PLQY"][index],
                            "total_parameters": total_parameters,
                            "encoding": encoding,
                            "molecule_name": molecule_name,
                            "monodispersity": self.synthesis_data["monodispersity"][index],
                            }
            

            self.data.append(data_point)


        # add the rest of the baseline data
        if self.add_baseline:
            self.data = self.add_limit_baseline(self.data)
            self.data = self.add_Toluene_baseline(self.data)
        

        # ....

        return self.data 
    




#### ----------------------------------  HELPERS  -------------------------------- ####


    def encode(self, molecule_name, encoding_type) -> list:

        """ Encodes the molecule name to a one hot or geometry encoding """


        # (1) this is a more complex encoding based on the molecule geometry
        if encoding_type == "geometry":

            encoding = None
            try:
                #  get entry from molecule_encoding.json
                encoding = self.molecule_geometry[molecule_name]

                # to list
                encoding = [float(x) for x in encoding]

            except KeyError:
                print(f"Geometry encoding not found for {molecule_name}")



        # (2) as the name suggests, this is just a one hot encoding based on the molecule name list
        elif encoding_type == "one_hot":   
            encoding = [0] * len(self.molecule_names)
            encoding[self.molecule_names.index(molecule_name)] = 1

        else:
            raise ValueError("No valid encoding specified")

        return encoding
    


    def read_spectrum(self, path, monodispersity = 1) -> tuple:

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

                # default is EV, later converted to nm if necessary
                wavelength.append(nm_to_ev(float(x)))     
                spectrum.append(float(y))


        """ TODO: both of the following operations could be done more elegantly
            both should ideally be evaluated by fitting a function, but since the
            data has very little noise, this is fine
        """ 

        # if monodispersity is set to 0, the peak position is evaluated from "centre of mass"
        if monodispersity == 0:
            if self.wavelength_unit == "EV":
                poly_peak_pos = np.average(wavelength, weights = spectrum)
            else:
                poly_peak_pos = ev_to_nm(np.average(np.array(wavelength), weights = spectrum))

        else: poly_peak_pos = -1


        # get peak position
        max_index = spectrum.index(max(spectrum))
        peak_pos  = wavelength[max_index]

        #logic for FWHM, could be changed
        half_max  = 0.5 * max(spectrum)

        # linear interpolation for the FWHM
        x_vec = np.linspace(wavelength[-1], wavelength[0], 10000)
        y_vec = np.interp(x_vec, np.flip(wavelength), np.flip(spectrum))

        # get indices for the left and right side of the peak
        above_half_max = y_vec > half_max
        left_index = np.where(above_half_max)[0][0]
        right_index = np.where(above_half_max)[0][-1]

        # FWHM
        fwhm = abs(x_vec[left_index] - x_vec[right_index])


        # position of the high energy shoulder (HM)
        if self.wavelength_unit == "EV":
            hm_high_energy = abs(x_vec[right_index] - x_vec[max_index])*2
        else:
            hm_high_energy = abs(x_vec[left_index] - x_vec[max_index])*2


        # if the mode is NM, convert the peak position to nm
        if self.wavelength_unit == "NM":
            peak_pos = ev_to_nm(peak_pos)
            wavelength = [ev_to_nm(x) for x in wavelength]

        # if the monodispersity is set to 0, the peak position is evaluated from "centre of mass"
        # every other quantity is NOT changed ---> remove for the paper
        if poly_peak_pos != -1:
            peak_pos = poly_peak_pos

        return fwhm, peak_pos, spectrum, wavelength, hm_high_energy



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




#### ------------------------------  SAVE TO FILE  ------------------------------- ####
    """         < in case the same selection is used multiple times >           """


    def save_data_as_file(self, data_objects, filename) -> None:
            
        """ Saves the data objects as a pickle file """

        filename += ".pkl"
        with open(filename, "wb") as filestream:
            pickle.dump(data_objects, filestream)


    def load_data_from_file(self, filename) -> list:

        """ Loads the data objects from a pickle file """

        filename += ".pkl"
        with open(filename, "rb") as filestream:
            data_objects = pickle.load(filestream)

        return data_objects




#### ------------------------------  init. helpers  ------------------------------ ####

    def read_global_attributes_and_normalize(self) -> pd.DataFrame:

        """ Reads in the global attributes of the molecules and normalizes them """

        # dataframe
        df = pd.read_csv( self.global_attributes_path, delimiter= ';', header= 0)

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

            n_Cs = df["c (Cs-OA)"] * df["V (Cs-OA)"]
            n_Pb = df["c (PbBr2)"] * df["V (PbBr2 prec.)"]
            

            # normalize the columns
            for key in synthesis_selection:
                df[key] = self.normalize(df[key], key)

            # create the synthesis data container
            synthesis_data = {key  : df[key].to_numpy().astype(float) for key in synthesis_selection} 

            # add ratios (probably the most expressive parameters)
            synthesis_data["AS_Pb_ratio"] = AS_Pb_ratio.to_numpy().astype(float)
            synthesis_data["Cs_Pb_ratio"] = Cs_Pb_ratio.to_numpy().astype(float)
            synthesis_data["Cs_As_ratio"] = Cs_As_ratio.to_numpy().astype(float)
            synthesis_data["n_Cs"] = n_Cs.to_numpy().astype(float)
            synthesis_data["n_Pb"] = n_Pb.to_numpy().astype(float)
            synthesis_data["n_As"] = np.zeros(len(n_Cs))

            # add target related data (seperate since they need different data typing)
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


    def get_molecule_distribution(self) -> dict:

        """ Get the distribution of the molecules in the synthesis data """

        distribution = {}
        for molecule in self.molecule_names:
            distribution[molecule] = len([data for data in self.data if data["molecule_name"] == molecule and not data["artificial"]])

        return distribution



#### --------------------------  BASELINE AUGMENTATION  -------------------------- ####

    def add_limit_baseline(self, data_objects):

        """ Adds a baseline to the data objects """
        print("Adding limit baseline")
    

        # get the 515nm baseline in an L-shape
        inputs  = [[i/10, 1] for i in range(0, 9)]
        #inputs += [[1, i/10] for i in range(6, 10)]
        
        peak  = 515
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
                                    "Cs_Pb_ratio": input[1], "AS_Pb_ratio": input[0],
                                    "peak_pos": peak, "index": -1, 
                                    "sample_number": "baseline",
                                    "baseline": True, "artificial": True,
                                    "monodispersity": 1,})

        return data_objects



    def add_Toluene_baseline(self, data_objects):

        """ Adds a baseline of the Toluene (no antisolvent) data to the data objects """
        print("Adding Toluene baseline")

        new_data_objects = data_objects.copy()

        # get Toluene data (the last constraint is a workaround to exclude the other baselines)
        toluene_data = [data for data in data_objects if data["molecule_name"] == "Toluene" and "AS_Pb_ratio" in data.keys()]


        # define the molecule selection
        if self.flags["molecule"] == "all":
            molecule_selection = self.molecule_names.copy()
            molecule_selection.remove("Toluene")
        else:
            molecule_selection = [self.flags["molecule"]]


        # add the Toluene data to the data objects
        for molecule in molecule_selection:
            for data in toluene_data:
                new_data_point = data.copy()
                new_data_point["molecule_name"] = molecule
                new_data_point["encoding"] = self.encode(molecule, self.encoding)
                new_data_point["baseline"] = True
                new_data_point["artificial"] = False
                new_data_objects.append(new_data_point)


        return new_data_objects




#### ----------------------------------  PLOTTING  -------------------------------- ####
    """            < don't put this on the final .git, it's a mess >              """


    def plot_data(self, var1=None, var2 = None, 
                  kernel = None, 
                  model = "GP", 
                  molecule = "all",
                  data_objects = None,
                  library = "plotly") -> None:

        """
            Scatter plot of the data in parameter space, for visualization;
            kernel can be plotted as well
        """


        # get the data

        if data_objects is None:
            data_objects = self.data

        Cs_Pb =      [data["Cs_Pb_ratio"] for data in data_objects
                 if data["molecule_name"] == molecule and not data["artificial"]]
        As_Pb =      [data["AS_Pb_ratio"] for data in data_objects
                 if data["molecule_name"] == molecule and not data["artificial"]]
        peak =       [data["peak_pos"] for data in data_objects
                 if data["molecule_name"] == molecule and not data["artificial"]]
        sample_no =  [data["sample_number"] for data in data_objects
                 if data["molecule_name"] == molecule and not data["artificial"]]
        print(sample_no)
        target =     [data["y"] for data in data_objects
                 if data["molecule_name"] == molecule and not data["artificial"]]

        target_base = [data["y"] for data in data_objects
                 if data["molecule_name"] == molecule and data["artificial"]]
        Cs_Pb_base = [data["Cs_Pb_ratio"] for data in data_objects
                    if data["molecule_name"] == molecule and data["artificial"]]
        As_Pb_base = [data["AS_Pb_ratio"] for data in data_objects
                    if data["molecule_name"] == molecule and data["artificial"]]
        peak_base = [data["y"] for data in data_objects
                    if data["molecule_name"] == molecule and data["artificial"]]



        # plot data with plotly
        if library == "plotly":
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects
                            .Scatter3d(x=As_Pb, y=Cs_Pb, z=peak, mode='markers', 
                                    marker=dict(size=13, opacity=0.7, color=peak,),
                                    text = sample_no,
                                    ))
            fig.update_traces(marker=dict(cmin=410, cmax=600, colorbar=dict(title='PEAK POS'), 
                                    colorscale='rainbow', color=peak, showscale=True, opacity=1),
                             textposition='top center')
            # fig.update_traces(marker=dict(cmin=0, cmax=80, colorbar=dict(title='PEAK POS'), 
            #                         colorscale='viridis', color=peak, showscale=True, opacity=1),
            #                  textposition='top center')
            fig.add_trace(plotly.graph_objects
                            .Scatter3d(x=As_Pb_base, y=Cs_Pb_base, z=target_base, mode='markers', 
                                        marker=dict(size=13, opacity=1, color="black"),
                                        ))
            fig.update_layout(scene = dict(xaxis_title=var1+ "[10^4]", 
                                    yaxis_title=var2 ,
                                    zaxis_title="Peak Position [nm]",),
                                    )

        elif library == "matplotlib":
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel("Peak Position [nm]")


        # plot model
        if kernel is not None:
            y_vec = np.linspace(0, 1, 100)
            x_vec = np.linspace(0, 0.8, 100)
            X, Y  = np.meshgrid(x_vec, y_vec)
            input = np.c_[X.ravel(), Y.ravel()]
            
            # add self.encode(molecule, self.encoding) to the input
            encoding = self.encode(molecule, self.encoding)
            input = [np.append(encoding, row ) for row in input]

            input = np.array(input)
            print(input.shape)
            Z = kernel.model.predict(input)[0].reshape(X.shape)
            err = kernel.model.predict(input)[1].reshape(X.shape)


            # write X, Y, Z to a csv with pandas
            #df = pd.DataFrame(data = Z, index = x_vec, columns = y_vec)
            #df.to_csv(f"model_{molecule}.csv")



            # add the surface plot of the kernel with a unifomr color
            if library == "plotly":
                fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=0.8, colorscale='greys', cmin = 200, cmax = 900))
                #fig.add_trace(plotly.graph_objects.Surface(x=x_vec, y=y_vec, z=Z, opacity=1, colorscale='Viridis', cmin = 430, cmax = 540))
            elif library == "matplotlib":
                ax.plot_surface(X, Y, Z, alpha=0.7, color = "gray", lw=0.5, rstride=8, cstride=8,)
                ax.contourf(X, Y, Z, zdir='z', offset=420, cmap='gist_rainbow_r', alpha=0.8, vmin = 410, vmax = 600, levels = 20)
                
                ax.scatter(As_Pb, Cs_Pb, peak, c = peak, cmap = "gist_rainbow_r", 
                           edgecolors='black', vmin = 410, vmax = 600, s = 100, alpha=1)
                ax.scatter(As_Pb_base, Cs_Pb_base,  peak_base, c = "black", edgecolors='black', s = 100, alpha=1)

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
        molecule_str = molecule
        if library == "plotly":
            #fig.write_html(f"plots/{molecule_str}.html")
            fig.show()
        
        elif library == "matplotlib":
            plt.show()

        return fig
        


    def plot_2D_contour(self, var1 = None, var2 = None, kernel = None) -> None:

        """
            2D contour plot of the data in parameter space, for visualization 
            purposes color coded by the target value (PEAK_POS)
        """

        # get the data
        x = [data["AS_Pb_ratio"] for data in self.data if not data["baseline"]]
        y = [data["Cs_Pb_ratio"] for data in self.data if not data["baseline"]]
        
        y_vec   = np.linspace(0, 1, 100)
        x_vec   = np.linspace(0, 1, 100)
        X, Y    = np.meshgrid(x_vec, y_vec)
        input   = np.c_[X.ravel(), Y.ravel()]

        # output string
        molecules = list(set([data["molecule_name"] for data in self.data if not data["baseline"]]))
        mol_str   = "_".join(molecules)

       


        # evaluate the kernel on the grid
        encoding = self.encode(molecules[0], self.encoding)
        input = np.array([np.append(encoding, row ) for row in input])

        if kernel is not None:
            print(input.shape)
            Z = kernel.model.predict(input)[0].reshape(X.shape)


            # a contour plot of the kernel with matplotlib
            # fig, ax = plt.subplots()
            #c = ax.contourf(X, Y, Z, 50, colors = "k", alpha = 0.4, linestyles = "dashed", xlim = (0, 1), ylim = (0, 1))
            #fig.colorbar(c, ax=ax, label = "PEAK POS")

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
        #c = [data["monodispersity"] for data in self.data if not data["baseline"]]
        c = [abs(data["y"]-480)*3 + 50*data["Cs_Pb_ratio"] for data in self.data if not data["baseline"]]
        # ax.scatter(x, y, c = c, s = 50,
        #            cmap = "bwr_r",
        #            edgecolors='black')
        #plt.show()

        # plot with plotly
        fig.add_trace(plotly.graph_objects.Scatter(x=x, y=y, mode='markers',
                                                    marker=dict(size=16, opacity=1, color = c, 
                                                                colorscale='hot', cmin=0, cmax=50,
                                                                showscale=True, colorbar=dict(title='FWHM [mev]]',
                                                                tickfont=dict(size=16),),
                                                                line=dict(width=1, color='Black'),
                                                                ),
                                                    ))
        # no ticks, no grid, no lines, no labels
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title_font = dict(size = 16))
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title_font = dict(size = 16))
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)



        # save the plot as svg and png
        fig.write_image(f"plots/Contour_Monodisp_{mol_str}.svg")
        fig.write_image(f"plots/Contour_Monodisp_{mol_str}.png")
        fig.show()


    def plot_2D_contour_old(self, kernel = None, molecule = None) -> None:

        """
            2D contour plot of the data in parameter space, for visualization 
            purposes color coded by the target value (PEAK_POS)
        """

        # get the data
        x = [data["total_parameters"][0] for data in self.data if data["molecule_name"] == molecule]
        y = [data["total_parameters"][1] for data in self.data if data["molecule_name"] == molecule]
        monodispersity = [data["monodispersity"] for data in self.data if data["molecule_name"] == molecule]
        color = [data["y"] for data in self.data if data["molecule_name"] == molecule]
        # x = [data["total_parameters"][0] for data in self.data if data["baseline"]]
        # y = [data["total_parameters"][1] for data in self.data if data["baseline"]]
        # monodispersity = [data["monodispersity"] for data in self.data if data["baseline"]]

        # output string
        molecules = list(set([data["molecule_name"] for data in self.data]))
        mol_str   = "_".join(molecules)


        # a contour plot of the kernel
        fig, ax = plt.subplots()
        y_vec   = np.linspace(0, 1, 100)
        x_vec   = np.linspace(0, 1, 100)
        X, Y    = np.meshgrid(x_vec, y_vec)
        input   = np.c_[X.ravel(), Y.ravel()]

        
        # evaluate the kernel on the grid
        encoding = self.encode(molecules[0], self.encoding)
        input = np.array([np.append(encoding, row ) for row in input])


        # evaluate the kernel on the grid
        if kernel is not None:
            print(input.shape)
            Z = kernel.model.predict(input)[0].reshape(X.shape)
            c = ax.contourf(X, Y, Z, 30, cmap='gist_rainbow_r', vmin = 400, vmax = 600, zorder = 1)

            # add black lines for the contour
            ax.contour(X, Y, Z, 30, colors='black', linewidths=0.5, zorder = 2)
            
            # colorbar with text size 12
            cbar = fig.colorbar(c, ax=ax, label = "PEAK POS",)
            cbar.ax.tick_params(labelsize=12,)
            cbar.set_label("PEAK POS", fontsize = 12)


        # plot the data
        ax.scatter(x, y, c = color, s = 80, vmin = 400, vmax = 600, 
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

        # save the plot as png and svg
        #plt.savefig(f"plots/Contour_{mol_str}.png")
        plt.savefig(f"plots/Contour_{mol_str}.svg")
        
        plt.show()


        # calculate the area below 464nm (3ML or less)

        area = 0
        for nm in Z:
            for val in nm:
                if val <= 464:
                    area += 1

        return area


    def plot_parameters(self, data_objects,) -> None:

        """
            Plots arbitrary parameters for different visualization purposes
            -> not pretty, but useful for quick checks
        """


        Cs_Pb_ratio =     	[data["Cs_Pb_ratio"] for data in data_objects]
        AS_Pb_ratio =       [data["AS_Pb_ratio"] for data in data_objects]
        Cs_As_ratio =       [data["Cs_As_ratio"] for data in data_objects]
        c_Perovskite =      [data["c_Perovskite"] for data in data_objects]
        enhanced =          [data["enhanced"] for data in data_objects]
        sample_no =         [data["sample_number"] for data in data_objects]
        target =            [data["y"]                   for data in data_objects]
        specific =          [1 if data["sample_number"] == "j19" else 0 for data in data_objects]
        _2ML_type =         [1 if data["peak_pos"] < 2.87 else 0 for data in data_objects]
        peak_pos =          [data["peak_pos"]            for data in data_objects]
        monodispersity =    [data["monodispersity"]      for data in data_objects]
        index =             [data["index"]               for data in data_objects]
        fwhm =              [round(data["fwhm"]*1000)                for data in data_objects]
        suggestion =        [1 if "L" in data["suggestion"] or "l" in data["suggestion"]
                             else 0 for data in data_objects]

        molecule_name =     [self.molecule_names.index(data["molecule_name"]) 
                             for data in data_objects]
        
        # lowest values for each peak position
        lowest_x, lowest_y = find_lowest(data_objects=data_objects)
        lowest_y = [y * 1000 for y in lowest_y]
        

        if len(Cs_Pb_ratio) == 0:
            raise ValueError(f"Not enough data found for molecule: {molecule_name}")

        
        """ basic scatter plot """
        fig, ax = plt.subplots(figsize = (3.5, 4))
        sign = -1 if self.wavelength_unit == "EV" else 1

        cbar = plt.colorbar(ax.scatter(peak_pos, target,
                                       c = Cs_Pb_ratio, cmap= "bwr_r", alpha = 1, s = 70,vmin = 0, vmax = 0.2)) # vmin = nm_to_ev(400), vmax = nm_to_ev(600)))
        
        # cbar = plt.colorbar(ax.scatter(peak_pos, target,
        #                                 c = peak_pos, cmap= "gist_rainbow_r", alpha = 1, s = 70, vmin = 400, vmax = 600))
        
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



    def plot_correlation(self) -> None:

        """
            Plots the correlation matrix of the data 
            together with the target values
        """

        # get the data
        data = [data["total_parameters"] for data in self.data]

        # create a dataframe, usefull for correlation matrix
        df = pd.DataFrame(data, columns = self.total_training_parameter_selection)
        df["plqy"]     = [data["plqy"] for data in self.data]
        df["fwhm"]     = [data["fwhm"] for data in self.data]
        df["peak_pos"] = [data["peak_pos"] for data in self.data]

        # plot the correlation matrix
        corr = df.corr()
        fig, ax = plt.subplots()
        im = ax.imshow(abs(corr), cmap="Blues")

        # --> the "+3" is used for the additional plqy, fwhm, peak_pos
        ax.set_xticks(np.arange(len(self.total_training_parameter_selection)+ 3))
        ax.set_yticks(np.arange(len(self.total_training_parameter_selection)+ 3))
        ax.set_xticklabels(self.total_training_parameter_selection 
                           + ["plqy", "fwhm", "peak_pos"], rotation='vertical')    
        ax.set_yticklabels(self.total_training_parameter_selection 
                           + ["plqy", "fwhm", "peak_pos"])


        plt.colorbar(im)
        plt.show()



    def plot_ternary(self, data_objects, kernel= None) -> None:

        """
            Ternary plot of the data in parameter space, for visualization
        """

        # amount of each substance
        Cs = np.array([data["amount_substance"]["Cs"] for data in data_objects])
        print(np.mean(Cs))
        Pb = np.array([data["amount_substance"]["Pb"]/4 for data in data_objects])
        print(np.mean(Pb))
        As = np.array([data["amount_substance"]["As"] for data in data_objects])
        print(np.mean(As))
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




#### --------------------------------  DICIONARIES  ------------------------------- ####


    def get_molecule_dictionary(self) -> dict:
        
        """ 
            Dictionaries for the molecule names and the NPL types 
            Should be moved to a separate file in the future, but this is fine for now
        """

        ml_dictionary = {#"1": (402, 407),
                         "2": (430, 437),
                         "3": (458, 464),
                         "4": (472, 481),
                         "5": (484, 489),
                         "6": (491, 497),
                         "7": (498, 504),
                         "8": (505, 509),
                         "9": (510, 520),}
        

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
                                "pentanone" : "Pentanone",
                                "3-Penon" : "3-Pentanone",
                                "PenOH" : "Pentanol",
                                }
        
        return molecule_dictionary, ml_dictionary


    def get_molecule_geometry(self, path = "data/molecule_encoding.json") -> dict:

        """ 
            Encodings for the molecules, should be moved to a separate file in the future
        """

        with open(path, "r") as file:
            encoding = json.load(file)

        print(encoding)

        return encoding
