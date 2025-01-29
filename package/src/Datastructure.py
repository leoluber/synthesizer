r""" Datastructure.py implements the Datastructure class used 
     throughout the project as main data structure  """
     # << github.com/leoluber >> 



from typing import Literal
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import pickle
import json
import os
from plotly import graph_objs as go
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


# custom
from src.helpers import nm_to_ev, ev_to_nm, find_lowest, surface_proportion




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
    - encoding:              encoding type for the molecules (one_hot, geometry)
    - wavelength_filter:     filter for the peak position (list of two values in nm)
    - molecule:              molecule to be selected (all, specific molecule)
    - add_baseline:          add a baseline data to the data objects
    - monodispersity_only:   exclude samples that are not monodisperse
    - P_only:                exclude samples that are not P type

    BASIC USAGE
    -------------
    >>> from Datastructure import Datastructure
    >>> ds = Datastructure(synthesis_file_path = "synthesis_data.csv", target = "FWHM", ...)
    >>> ds.get_training_data(["Cs_Pb_ratio", "AS_Pb_ratio"], "FWHM")



    FEATURES
    ------------
    Can be found in the example data file

    """




    def __init__(
        self,
        synthesis_file_path:   str, 
        spectral_file_path:   str = "spectrum/",
        encoding:        Literal["one_hot", "geometry",] = "one_hot",
        wavelength_filter =    [400, 800],
        molecule =            "all",
        add_baseline =         False,
        monodispersity_only =  False,
        P_only =               False,
        ):
        

        # main stettings
        self.encoding        = encoding
        self.add_baseline    = add_baseline
        self.wavelength_filter = wavelength_filter


        # selection flags
        self.flags = {
                      "monodispersity_only": monodispersity_only,
                      "P_only"             : P_only,
                      "molecule"           : molecule}
        

        # directories
        self.current_path =             os.getcwd()
        self.data_path_raw =            self.current_path + "/data/raw/"
        self.data_path_processed =      self.current_path + "/data/processed/"
        self.synthesis_file_path =      self.data_path_raw + synthesis_file_path
        self.global_attributes_path =   self.data_path_processed + "AntisolventProperties.csv"
        self.spectrum_path =            self.data_path_raw + spectral_file_path
        self.geometry_path=             self.data_path_raw + "molecule_encoding.json"
        self.molecule_dictionary_path = self.data_path_raw + "molecule_dictionary.json"
        self.ml_dictionary_path =       self.data_path_raw + "ml_dictionary.json"



        # dictionaries for molecule names, geometries and atom numbers
        self.molecule_dictionary, self.ml_dictionary, self.molecule_geometry = self.get_dictionaries()


        # list of all relevant molecule names
        self.molecule_names = list(self.molecule_dictionary.values())


        # for normalization
        self.max_min = {}  # norm/denorm


        # molecule attributes
        self.global_attributes_df =  pd.read_csv(self.global_attributes_path, 
                                                 delimiter= ';', header= 0)


        # densities for adjusting the concentrations
        self.densities = {f"{molecule}": self.global_attributes_df
                          .loc[self.global_attributes_df['antisolvent'] == molecule]['n [mol/L]']
                          .to_numpy()
                          .astype(float)[0] for molecule in self.molecule_names}
        

        # read the synthesis data
        self.read_synthesis_data()




#### ------------------------------  READING DATA ------------------------------ ####


    def read_synthesis_data(self) -> dict:

        """ Read in the synthesis data from dataframe and return as container (dict.) 
        
        The synthesis data is read in from the synthesis file and normalized.
        The data is then stored in a dictionary container:
        - the keys are the column names of the synthesis data file
        - the values are the normalized data as numpy arrays
        --> ratios are NOT normalized, use units instead to rescale them
        """

        print("Reading synthesis data...")


        # define the essential and non-essential parameters and the parameters to normalize
        essential_parameters = ["Sample No.", "c (PbBr2)", "c (Cs-OA)" , "V (Cs-OA)","V (antisolvent)", "V (PbBr2 prec.)", "antisolvent"]
        non_essential_parameters = ["PLQY", "monodispersity", "Reference", "PL_data", "Pb/I", "Age of prec", "S/P", "suggestion", "NC shape"]
        parameters_to_normalize = ["n_Cs", "n_Pb", "V_total", "V (Cs-OA)", "V (PbBr2 prec.)", "V (antisolvent)", "c (PbBr2)", "c (Cs-OA)"]


        # read the dataframe from the defined path
        if not os.path.exists(self.synthesis_file_path): 
            raise FileNotFoundError(f"File not found: {self.synthesis_file_path}")
        df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)  


        #check if the critical columns are present
        for key in essential_parameters:
            if key not in df.columns:
                raise KeyError(f"KeyError: {key} not found in synthesis data")
            
        
        # print out unfamiliar columns
        print("Unfamiliar columns:")
        unfamiliar_columns = [key for key in df.columns if key not in essential_parameters + non_essential_parameters]
        print(unfamiliar_columns)
        

        # remove unwanted molecules  TODO: move this to the data selection
        recognized_molecules = list(self.molecule_dictionary.keys())
        df['antisolvent'] = df['antisolvent'].replace("0", "Tol")
        df = df[df['antisolvent'].isin(recognized_molecules)]

        print(df)

        # calculate properties 
        # (if these raise an error, there is a fundamental issue with the provided dataset)
        try:
            print(len(df["c (PbBr2)"]), len(df["V (PbBr2 prec.)"]))
            df['n_Pb'] = (df["c (PbBr2)"]* df["V (PbBr2 prec.)"]),
            df['Cs_Pb_ratio'] = (df["n_Cs"] / df["n_Pb"]),
            df['AS_Pb_ratio'] = (df["V (antisolvent)"]) / (df["V (PbBr2 prec.)"] * df["c (PbBr2)"]),
            df['Cs_As_ratio'] = (df["c (Cs-OA)"] * df["V (Cs-OA)"]) / (df["V (antisolvent)"] + 0.0001),
            df['V_total'] = df["V (Cs-OA)"] + df["V (antisolvent)"] + df["V (PbBr2 prec.)"],

        except KeyError:
            print("KeyError: central properties not found in synthesis data")
            return None


        # normalize properties 
        for key in parameters_to_normalize:
            df[key] = self.normalize(df[key], key)




        ### ----------------------  CASE SPECIFIC HANDELING  ---------------------- ###
        # if these raise an error, there are workarounds in place to handle the issue
        # TODO: generalize this, it's a mess
        

        """ >>> SYNTHESIS TIME:  normalize the , log to handle the large range """
        try:
            df["t_RKT"] = self.normalize(np.log(df["t_RKT"]))
        except KeyError:
            print("KeyError: t_RKT not found in synthesis data")
            df["t_RKT"] = np.zeros(len(df["Sample No."]))


        """ >>> MONODISPERSITY: can be a either a boolean or a float """
        try:
            if set(df["monodispersity"]) <= {0, 1, ""}:
                df["monodispersity"] = df["monodispersity"].replace("", 0)
                df["monodispersity"] = df["monodispersity"].astype(int)
            else:
                df["monodispersity"] = self.normalize(df["monodispersity"], "monodispersity")

        except KeyError:
            print("KeyError: monodispersity not found in synthesis data")
            df["monodispersity"] = np.ones(len(df["Sample No."]))

        
        """ >>>  PL_data: if not found, use the 'sample_number.txt' instead """
        try:
            if "Pl_data" not in df.columns:
                df["PL_data"] = df["Sample No."].astype(str) + ".txt"
            else:
                df["PL_data"] = df["PL_data"].astype(str)

        except KeyError:
            print("KeyError: PL_data not found in synthesis data")
            df["PL_data"] = df["Sample No."].astype(str) + ".txt"
        

        """ >>> suggestion: if not found, use an empty string instead """
        if "suggestion" not in df.columns:
            df["suggestion"] = ""

        """ >>> molecule_name """
        df["molecule_name"] = df["antisolvent"]
        for index, molecule in enumerate(df["molecule_name"]):
            df["molecule_name"][index] = self.molecule_dictionary[molecule]


        ### -------------------------  PREPROCESSING  -------------------------- ###

        # classify the product
        df["NPL_type"] = [self.get_NPL_type(peak_pos) for peak_pos in df["peak_pos"]]


        # add empty columns for parameters needed at a later stage
        for key in ["baseline", "artificial", ]:
            df[key] = np.zeros(len(df["Sample No."], dtype = bool))


        # encoding
        df["encoding"] = [self.encode(self.molecule_dictionary[molecule]) 
                        for molecule in df["molecule_names"]]


        # correct ratios by antisolvent density
        df["n_As"] = np.zeros(len(df["Sample No."]))

        for index, molecule in enumerate(df["molecule_names"]):
            df['AS_Pb_ratio'][index] = df['AS_Pb_ratio'][index] * self.densities[molecule] / 10000
            df['n_As'][index] = df['V (antisolvent)'][index] * self.densities[molecule]


        
        # read spectral data
        for index, sample_number in enumerate(df["Sample No."]):
            try:
                path = self.spectrum_path + df["PL_data"][index]

            except KeyError:
                print(f"KeyError: PL_data column not in synthesis data, using sample number instead")
                path = self.spectrum_path + str(sample_number)
            
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            fwhm, peak_pos, poly_peak_pos, peak_pos_eV = self.read_spectrum(path, monodispersity = df["monodispersity"][index])

            df["fwhm"][index] = fwhm
            df["peak_pos"][index] = peak_pos
            df["poly_peak_pos"][index] = poly_peak_pos
            df["peak_pos_eV"][index] = peak_pos_eV


        # add baseline data
        if self.add_baseline:
            df = self.add_limit_baseline(df)
            df = self.add_Toluene_baseline(df)


        # write updated dataframe to csv
        df.to_csv("data/processed/data_processed.csv", sep = ";")

        print("Data read and processed successfully")

        return 0


    def get_training_data(self, training_selection: list, target: str) -> tuple:

        """ reads processed_data.csv and returns the training data as a numpy arrays 
            
            USAGE: after the data is read in and processed through read_synthesis_data()
            the training data can be generated using this function
        """

        # read the processed data
        data_frame = pd.read_csv("data/processed/data_processed.csv", delimiter= ";", header= 0)


        #### --------------- TEMPORARY CHANGES --------------- ####
        data_frame = data_frame[data_frame["Sample No."] > 120]


        # check if the keys are in the dataframe
        for key in (training_selection + [target]):
            if key not in data_frame.columns:
                raise KeyError(f"KeyError: {key} not found in synthesis data")
            

        # make sample selection based on self.flags and other criteria
        data_frame = data_frame[data_frame['Cs_Pb_ratio'] <= 1]
        data_frame = data_frame[data_frame['peak_pos'] <= max(self.wavelength_filter)]
        data_frame = data_frame[data_frame['peak_pos'] >= min(self.wavelength_filter)]

        if self.flags["molecule"] != "all":
            data_frame = data_frame[data_frame['molecule_name'] == self.flags["molecule"]]
        if self.flags["monodispersity_only"]:
            data_frame = data_frame[int(data_frame['monodispersity']) == 1]
        if self.flags["P_only"]:
            data_frame = data_frame[data_frame['S/P'] == "P"]
        if target == "PLQY":
            data_frame = data_frame[data_frame['PLQY'] != 0]
        

        # get the training data
        x = data_frame[training_selection].to_numpy()
        y = data_frame[target].to_numpy()


        return x, y
            


#### ----------------------------------  HELPERS  -------------------------------- ####


    def encode(self, molecule_name,) -> list:

        """ Encodes the molecule name to a geometry encoding """

        try:
            geo_encoding = [float(x) for x in self.molecule_geometry[molecule_name]]

        except KeyError:
            print(f"KeyError: {molecule_name} not found in molecule dictionary")
            return None

        return geo_encoding



    def read_spectrum(self, path,) -> tuple:

        """ Reads in the spectral data from a .txt or .csv file
        
        Minimum requirements for the file:
        - 2 or 3 columns
        - header no longer than 4 lines (those are skipped)
        - the data is comma separated !!!
        - the first column is the wavelength (nm)
        - the last column is the intensity 
        """
    
        energies, spectrum = [], []
        with open(path, "r") as filestream:
            for i, line in enumerate(filestream):

                # skip the header lines and empty lines
                row = line.split(",")
                try:
                    float(row[0])
                except ValueError:
                    continue

                # read the data
                if len(row) == 2:
                    x, y = row
                elif len(row) == 3:
                    x, _, y = row
                else:
                    print(f"Invalid data format in {path}")
                    return None

                # default is EV, later converted to nm if necessary
                energies.append(nm_to_ev(float(x)))     
                spectrum.append(float(y))


        # normalize the spectrum
        spectrum = [x/max(spectrum) for x in spectrum]


        ### -- calulate two different peak positions -- ###
        # poly_peak_pos is the peak position calculated from the "centre of mass"
        max_index = spectrum.index(max(spectrum))
        peak_pos = energies[max_index]
        peak_pos_eV = peak_pos.copy()
        poly_peak_pos = np.average(np.array(energies), weights = spectrum)


        ### -- calculate the FWHM -- ###
        # linear interpolation for the FWHM
        x_vec = np.linspace(wavelength[-1], wavelength[0], 10000)
        y_vec = np.interp(x_vec, np.flip(wavelength), np.flip(spectrum))

        # get indices for the left and right side of the peak
        above_half_max = y_vec > (0.5 * max(spectrum))
        left_index = np.where(above_half_max)[0][0]
        right_index = np.where(above_half_max)[0][-1]

        # FWHM
        fwhm = abs(x_vec[left_index] - x_vec[right_index])


        # if the mode is NM, convert the peak position to nm
        if self.wavelength_unit == "NM":
            peak_pos = ev_to_nm(peak_pos)
            wavelength = [ev_to_nm(x) for x in wavelength]

        return fwhm, peak_pos, poly_peak_pos, peak_pos_eV



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




#### ----------------------------  BASELINE ADDITION  --------------------------- ####

    def add_limit_baseline(self, dataframe, molecules: list):

        """ Adds a number of extrapolated data points to the dataframe 
            based on physical and chemical insights
        """
    
        # define the inputs and the peak position
        inputs  = [[i/10, 1] for i in range(0, 1)]
        peak  = 515

        if self.wavelength_unit == "EV":
            peaks = [nm_to_ev(peak) for i in range(len(inputs))]
        else:
            peaks = [peak for i in range(len(inputs))]


        # add new data to data objects
        for molecule in molecules:
            for i, input_ in enumerate(inputs):
                new_row = {}
                new_row["Sample No."] = "baseline"
                new_row["molecule_name"] = molecule
                new_row["baseline"] = True
                new_row["artificial"] = True
                new_row["encoding"] = self.encode(molecule, self.encoding)
                new_row["peak_pos"] = peaks[i]
                new_row["fwhm"] = 0
                dataframe.append(new_row)

        return dataframe



    def add_Toluene_baseline(self, dataframe, molecules: list):

        """ Adds a baseline of the Toluene (no antisolvent) data to all antisolvents"""
        print("Adding Toluene baseline")

        # we don't want to add the Toluene data to the Toluene data
        molecules.drop("Toluene")

        # get Toluene data (excluding the limit baselines)
        toluene_data = dataframe[dataframe["molecule_name"] == "Toluene" and dataframe["baseline"] == False]

        # iterate the rows of tolune data and add the baseline to the other molecules
        for index, row in toluene_data.iterrows():
            for molecule in molecules:
                new_row = row.copy()
                new_row["molecule_name"] = molecule
                new_row["baseline"] = True
                new_row["Sample No."] = "baseline"
                new_row["artificial"] = False
                new_row["fwhm"] = 0
                new_row["encoding"] = self.encode(molecule, self.encoding)
                dataframe.append(new_row)
            
        return dataframe



        return new_data_objects




#### --------------------------------  DICIONARIES  ------------------------------- ####

    def get_dictionaries(self) -> dict:
        
        """ 
            Read the molecule dictionary and ML dictionary from the json files
        """

        # read the molecule dictionary
        try:
            with open(self.molecule_dictionary_path, "r") as file:
                molecule_dictionary = json.load(file)
            
            with open(self.ml_dictionary_path, "r") as file:
                ml_dictionary = json.load(file)

            with open(self.geometry_path, "r") as file:
                encoding = json.load(file)

            return molecule_dictionary, ml_dictionary, encoding
        
        except FileNotFoundError:
            print("Could not find the molecule dictionary, encoding dictionary or the ML dictionary file. "+
                   "Please make sure the files are in the correct path and in .json format.")
            return None
