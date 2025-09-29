""" 
    Project:     synthesizer
    File:        Datastructure.py
    Description: Defines the Datastructure class, which reads in the synthesis 
                 data, spectral information and adds preprocessing steps
    Author:      << github.com/leoluber >> 
    License:     MIT
"""





import numpy as np
import os
import pandas as pd
import json
import os
import sys
from typing import Literal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.helpers import nm_to_ev, ev_to_nm





class Datastructure:


    """ Datastructure class for the Synthesizer project

    General purpose data structure that reads in the synthesis data, spectral information 
    and the global attributes of the molecules. It performs various selections and saves 
    the data in a standardized .csv file that can be used for machine learning tasks.
    Can be easily extended to include more attributes or other target values.


    DATA REQUIREMENTS
    -----------------
    - all experimental data should be in "/data/raw/" and contain:
        - a folder "spectrum" with the spectral data in .txt or .csv files 
        - a .csv file with the synthesis data containing the sample numbers, molecule names, 
          synthesis parameters and other properties
        - a .csv file with the global attributes of the molecules (AntisolventProperties.csv)
        - a .json file with the molecule dictionary (molecule_dictionary.json)
        - a .json file with the molecule encodings (molecule_encoding.json)
        - a .json file with the monolayer dictionary (ml_dictionary.json)

        
    ARGS
    ----
    - synthesis_file_path (str):   path to the synthesis data file (.csv)
    - spectral_file_path (str):    path to the spectral data folder containing (txt, csv)
    - wavelength_filter (tuple):   filter for the peak position (list of two values in nm)
    - molecule (str):              molecule to be selected ("all" or  specific molecule)
    - add_baseline (bool):         add a baseline data to the data objects
    - monodispersity_only (bool):  exclude samples that are not monodisperse
    - P_only (bool):               exclude samples that are not P type (precipitation)

    
    BASIC USAGE
    -------------
    >>> from package.src.Datastructure import Datastructure
    >>> ds = Datastructure(synthesis_file_path = "synthesis_data.csv", ...)
    >>> ds.read_synthesis_data()
    >>> inputs, targets, selection_dataframe = datastructure.get_training_data(training_selection=features, 
                                                                               target="peak_pos", encoding=True)
    
    """



    def __init__(self,
        synthesis_file_path:   str, 
        spectral_file_path:    str = "spectrum/",
        wavelength_filter =    [400, 800], #nm
        molecule =             "all",
        encoding =             "geometry", # "geometry" or "chemical", "combined"
        add_baseline =         False,
        monodispersity_only =  False,
        P_only =               False,
        S_only =               False,
        fitting =              False,
        ):
        
        # check S/P
        if S_only and P_only:
            raise ValueError("S_only and P_only cannot be True at the same time")
        

        # main stettings
        self.add_baseline      = add_baseline
        self.wavelength_filter = wavelength_filter
        self.encoding          = encoding

        # selection flags
        self.flags = {"monodispersity_only": monodispersity_only,
                      "P_only"             : P_only,
                      "S_only"             : S_only,
                      "molecule"           : molecule}
        
        
        ##### --------------- TODO: IMPLEMENT YOUR FEATURE NAMES HERE --------------- ####
        # feature names (essential and non-essential)
        self.essential_parameters     = ["Sample No.", "c (PbBr2)", "c (Cs-OA)" , 
                                         "V (Cs-OA)","V (antisolvent)", "V (PbBr2 prec.)", 
                                         "antisolvent", "S/P", "monodispersity",]
        
        self.parameters_to_normalize  = ["V_total", "V (Cs-OA)", "V (PbBr2 prec.)", "V (antisolvent)", "c (PbBr2)", 
                                        "c (Cs-OA)",]
        ##### ----------------------------------------------------------------------- ####


        # directories
        self.dataset =                  synthesis_file_path
        self.current_path =             os.getcwd()   # needs to be changed when using linux
        self.data_path_raw =            self.current_path  + "/data/raw/"
        self.data_path_processed =      self.current_path  + "/data/processed/"
        self.synthesis_file_path =      self.data_path_raw + synthesis_file_path
        self.processed_file_path =      self.data_path_processed + "processed_" + synthesis_file_path
        self.global_attributes_path =   self.data_path_raw + "AntisolventProperties.csv"
        self.spectrum_path =            self.data_path_raw + spectral_file_path
        self.encoding_path=             self.data_path_raw + "molecule_encoding.json"
        self.molecule_dictionary_path = self.data_path_raw + "molecule_dictionary.json"
        self.ml_dictionary_path =       self.data_path_raw + "ml_dictionary.json"



        # dictionaries for molecule names, geometries and atom numbers
        self.molecule_dictionary, self.ml_dictionary, self.encoding_dictionary =  self.get_dictionaries()


        # list of all relevant molecule names
        self.molecule_names = list(self.molecule_dictionary.values())

        # for normalization and denormalization
        self.max_min = {}

        # molecule attributes
        self.global_attributes_df =  pd.read_csv(self.global_attributes_path, 
                                                 delimiter= ';', header= 0)
        
        # read the antisolvent densities
        self.densities = {f"{molecule}": self.global_attributes_df
                          .loc[self.global_attributes_df['antisolvent'] == molecule]['n [mol/L]']
                          .to_numpy()
                          .astype(float)[0] for molecule in self.molecule_names}
        




#### ----------------------------  READING & PREPROCESSING --------------------------- ####


    def read_synthesis_data(self):

        """ Read in the synthesis data from dataframe and return as container (dict.) 
        
        The synthesis data is read in from the synthesis file and normalized.
        The data is then stored in a dictionary container:
        - the keys are the column names of the synthesis data file
        - the values are the normalized data as numpy arrays
        --> ratios are NOT normalized, use units instead to rescale them

        """

        print("Reading synthesis data...")

        # get initial dataframe
        df = self.get_synthesis_dataframe()

        # calculate the essential properties
        df = self.calculate_properties(df)

        # normalize properties 
        for key in self.parameters_to_normalize:
            try:
                df[key] = self.normalize(df[key], key)
            except KeyError:
                print(f"KeyError: {key} not found in synthesis data")
                continue

        # handle specific properties
        df = self.process_additional_properties(df)

        # add spectral data
        df = self.add_spectral_data(df)

        # add baseline data
        if self.add_baseline:
            df = self.add_limit_baseline(df)
            df = self.add_Toluene_baseline(df)

        # write updated dataframe to csv
        df.to_csv(self.processed_file_path, sep = ";")

        # write meta data to file
        df_meta_data = pd.DataFrame.from_dict(self.max_min, orient = "index", columns = ["max", "min"])
        df_meta_data.to_csv(self.data_path_processed + "meta_data_" + self.dataset, sep = ";")

        print("Data read and processed successfully!")
        return 0
    

    def get_synthesis_dataframe(self) -> pd.DataFrame:
        
        """ Reads the synthesis data from the csv file 

        The function reads the synthesis data from the csv file and checks if the
        essential parameters are present.

        RETURNS
        -------
        - df (pd.DataFrame): the synthesis data

        """

        # read the dataframe from the defined path
        if not os.path.exists(self.synthesis_file_path):
            raise FileNotFoundError(f"File not found: {self.synthesis_file_path}")
        df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)  

        #check if the critical columns are present
        for key in self.essential_parameters:
            if key not in df.columns:
                raise KeyError(f"KeyError: {key} parameter not found in synthesis data")

        # remove unknown molecules and replace "0" with "Toluene"
        recognized_molecules = list(self.molecule_dictionary.keys())
        df['antisolvent'] = df['antisolvent'].replace("0", "Tol")
        df = df[df['antisolvent'].isin(recognized_molecules)]

        return df


    def calculate_properties(self, df) -> pd.DataFrame:

        """ Calculate the essential properties from the data
            
        The function calculates the essential properties from the synthesis data
        and adds them to the dataframe. (if this throws an error, there is an inherent
        issue with the data)

        ARGS
        ----
        - df (pd.DataFrame): the synthesis data

        RETURNS
        -------
        - df (pd.DataFrame): the updated synthesis data

        """

        ### --------------- TODO: SPECIFY YOUR CACULATED PROPERTIES HERE --------------- ###

        try:
            df["n_Cs"] = df["c (Cs-OA)"] * df["V (Cs-OA)"]
            df["n_Pb"] = df["c (PbBr2)"] * df["V (PbBr2 prec.)"]
            df["n_As"] = df["V (antisolvent)"]
            df['Cs_Pb_ratio'] = df["n_Cs"] / df["n_Pb"]
            df["AS_Pb_ratio"] = df["n_As"] / (df["n_Pb"] * 10000)
            df["AS_Cs_ratio"] = df["n_As"] / (df["n_Cs"] * 10000)
            df["V (antisolvent)"] = df["V (antisolvent)"]
            df['V_total'] = df["V (Cs-OA)"] + df["V (antisolvent)"] + df["V (PbBr2 prec.)"]

        ### --------------------------------------------------------------------------- ###

        except KeyError:
            print("KeyError: central properties not found in synthesis data")
            return None

        return df
    

    def process_additional_properties(self, df) -> pd.DataFrame:

        """Handles specific properties of the synthesis data
            
        (especially the ones that are not found in all datasets);
        if these raise an error, there are workarounds in place to handle the issue

        ARGS
        ----
        - df (pd.DataFrame): the synthesis data

        RETURNS
        -------
        - df (pd.DataFrame): the updated synthesis data

        """
    

        df["PL_data"] = [str(x) + ".txt" for x in df["Sample No."]]
        df["molecule_name"] = np.array([self.molecule_dictionary[molecule] for molecule in df["antisolvent"]])

        
        # addditional columns: prepare the dataframe for later use
        # will be used later to mark the extrapolated data
        for key in ["baseline", "artificial", ]:
            df[key] = np.zeros(len(df["Sample No."]), dtype = bool)

        # encoding: geometry encoding or chemical encoding
        df["encoding"] = [self.encode(molecule)
                        for molecule in df["molecule_name"]]

        # Antisolvent correction: calculate the AS_Pb_ratio and n_As by considering the density and units
        df["n_As"] = np.zeros(len(df["Sample No."]))
        for index, row in df.iterrows():
            df["n_As"][index] =  row["V (antisolvent)"] * self.densities[row["molecule_name"]]
            df["AS_Pb_ratio"][index] = row["AS_Pb_ratio"] * self.densities[row["molecule_name"]]


        return df


    def add_spectral_data(self, df) -> pd.DataFrame:

        """ Read the spectral data from external files
        """

        # define the columns for the spectral data
        fitting_properties = ["fwhm", "peak_pos",]
        for key in fitting_properties:
            df[key] = np.zeros(len(df["Sample No."]))

        # iterate the rows and read the spectral data
        print("Reading spectral data...")
        for index, row in df.iterrows():
            try:
                fitting_dictionary = \
                    self.read_spectrum(self.spectrum_path + row["PL_data"])
                
                for key in fitting_properties:
                    df[key][index] = fitting_dictionary[key]

            except TypeError:
                print(f"TypeError: {row['PL_data']} not found in spectral data")
                continue
        print("Spectral data read successfully!")

        plt.show()

        # remove all rows with missing spectral data
        df = df[df["peak_pos"] != 0]

        # classify the product
        df["NPL_type"] = [self.get_NPL_type(peak_pos) for peak_pos in df["peak_pos"]]

        return df



#### ------------------------------  TRAINING DATA  ------------------------------- ####


    def get_training_data(self, training_selection: list, 
                          target: str, encoding = False, remove_baseline = False )-> tuple:

        """ Reads processed_data.csv and returns the training data as a numpy arrays 
            
        USAGE
        -----

        After the data is read in and processed through read_synthesis_data()
        the training data can be generated using this function

        ARGS
        ----
        - training_selection (list): list of keys to be used as training data
        - target (str): target value to be predicted

        RETURNS
        -------
        - x (np.array): training data
        - y (np.array): target data
        - data_frame (pd.DataFrame): the processed dataframe

        """

        # read the processed data
        data_frame = pd.read_csv(self.processed_file_path, delimiter= ";", header= 0)

        # remove the baseline data if requested
        if remove_baseline:
            data_frame = data_frame[data_frame["baseline"] == False]

        # check if the keys are in the dataframe
        for key in (training_selection + [target]):
            if key not in data_frame.columns:
                raise KeyError(f"KeyError: {key} not found in synthesis data")

        # make sample selection based on self.flags and other criteria
        data_frame = data_frame[data_frame['peak_pos'] <= max(self.wavelength_filter)]
        data_frame = data_frame[data_frame['peak_pos'] >= min(self.wavelength_filter)]

        # filter the data based on the flags
        if self.flags["molecule"] != "all":
            data_frame = data_frame[data_frame['molecule_name'] == self.flags["molecule"]]

        if self.flags["monodispersity_only"]:
            print("Removing non-monodisperse samples")
            print(len(data_frame[data_frame['monodispersity'] != 0]))
            print(len(data_frame))
            data_frame = data_frame[data_frame['monodispersity'] != 0]
            print(len(data_frame))

        if self.flags["P_only"]:
            print("Removing non-P type samples")
            data_frame = data_frame[data_frame['S/P'] != "S"]
        
        elif self.flags["S_only"]:
            print("Removing non-S type samples")
            data_frame = data_frame[data_frame['S/P'] != "P"]

        if target == "PLQY":
            data_frame = data_frame[data_frame['PLQY'] != 0]
            data_frame = data_frame.dropna(subset = ["PLQY"])

        # get the training data
        x = data_frame[training_selection].to_numpy()
        y = data_frame[target].to_numpy()

        # add the encoding if requested
        if encoding:
            encodings = np.array([json.loads(x) for x in data_frame["encoding"]])
            x = np.concatenate((encodings, x), axis = 1)

        return x, y, data_frame,



#### ----------------------------------  HELPERS  -------------------------------- ####


    def encode(self, molecule_name, enc = None) -> list:

        """ Encodes the molecule name to a geometry encoding 
        
        RETURNS
        -------
        - geo_encoding (list): the geometry encoding of the molecule """


        if enc is None:
            enc = self.encoding


        match enc:
            case "chemical":
                selection = ["relative polarity (-)", "dielectric constant (-)","dipole moment (D)",
                            "Hansen parameter hydrogen bonding (MPa)1/2","Gutman donor number (kcal/mol)"]
                global_attributes = self.global_attributes_df.loc[self.global_attributes_df['antisolvent'] == molecule_name]

                return list(global_attributes[selection].to_numpy().astype(float)[0])
            
            case "geometry":

                return [float(x) for x in self.encoding_dictionary[molecule_name]]

            case "combined":
                geo_encoding = [float(x) for x in self.encoding_dictionary[molecule_name]]
                selection = ["relative polarity (-)", "Hansen parameter hydrogen bonding (MPa)1/2",]
                global_attributes = self.global_attributes_df.loc[self.global_attributes_df['antisolvent'] == molecule_name]
                chem_encoding = list(global_attributes[selection].to_numpy().astype(float)[0])

                return geo_encoding + chem_encoding
            
            case "strength":
                
                selection = ["relative polarity (-)",]
                global_attributes = self.global_attributes_df.loc[self.global_attributes_df['antisolvent'] == molecule_name]
                chem_encoding = list(global_attributes[selection].to_numpy().astype(float)[0])

                length = self.molecule_geometry[molecule_name][0]
                cone_angle = self.molecule_geometry[molecule_name][1]
                strength = self.molecule_geometry[molecule_name][2]

                return [strength, cone_angle, length, ] + chem_encoding
            
            case _:
                print("Invalid encoding type")


    def read_spectrum(self, path,) -> dict:

        """ Reads in the spectral data from a .txt or .csv file
        
        Minimum requirements for the file:
        - 2 or 3 columns
        - header no longer than 4 lines (those are skipped)
        - the data is comma separated !!!
        - the first column is the wavelength (nm)
        - the last column is the intensity 
        """

        sample_name = os.path.basename(path).split(".")[0]

        if not os.path.exists(path):
            return 0,0,0,0,0,0,0
    
        # read the data from the file
        energies, spectrum, wavelengths = [], [], []
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
                wavelengths.append(float(x))
                energies.append(nm_to_ev(float(x))) 
                spectrum.append(float(y))



        # normalize the spectrum
        spectrum = [x/max(spectrum) for x in spectrum]


        ### -- calulate two different peak positions -- ###
        # poly_peak_pos is the peak position calculated from the "centre of mass"
        max_index = spectrum.index(max(spectrum))
        peak_pos = energies[max_index]

        ### -- calculate the FWHM -- ###
        # linear interpolation for the FWHM
        x_vec = np.linspace(energies[-1], energies[0], 10000)
        y_vec = np.interp(x_vec, np.flip(energies), np.flip(spectrum))

        # get indices for the left and right side of the peak
        above_half_max = y_vec > (0.5 * max(spectrum))
        left_index = np.where(above_half_max)[0][0]
        right_index = np.where(above_half_max)[0][-1]

        # FWHM
        fwhm = abs(x_vec[left_index] - x_vec[right_index])

        # convert the peak position to eV
        peak_pos = ev_to_nm(peak_pos)
        wavelengths = [ev_to_nm(x) for x in energies]
        
        dictionary = {"peak_pos": peak_pos, "fwhm": fwhm}
    
        return dictionary


    def normalize(self, a, name):

        """ Norm. dataframes or arrays and stores the max and min values for denormalization 
        
        ARGS
        ----
        - a (np.array or pd.dataframe): the data to be normalized
        - name (str): the name of the data to be normalized

        RETURNS
        -------
        - a (np.array or pd.dataframe): the normalized data

        """

        self.max_min[name] = [a.max(), a.min()]

        if (a.max() - a.min()) == 0: 
            print(f"Normalization issue: {a}  has no range; returning original array")
            return a

        return (a-a.min())/(a.max()-a.min())
    

    def denormalize(self, a, name):

        """ Denormalizes a value based on the max and min values 
        
        ARGS
        ----
        - a (float): the value to be denormalized
        - name (str): the name of the data to be denormalized

        RETURNS
        -------
        - a (float): the denormalized value
        
        """

        try:
            max_val, min_val = self.max_min[name]
        except KeyError:
            print(f"KeyError: {name} not found in max_min dictionary")

        return a * (max_val - min_val) + min_val


    def get_NPL_type(self, peak_pos) -> float:

        """ Classify the NPL type from the peak position using the ml_dictionary 
        
        ARGS
        ----
        - peak_pos (float): the peak position in nm

        RETURNS
        -------
        - NPL_type (float): the NPL type (2-12) of the peak position

        """

        for key, value in self.ml_dictionary.items():
            if value[0] <= peak_pos <= value[1]:
                return float(key)
            
        return 0.



#### ----------------------------  BASELINE ADDITION  --------------------------- ####

    def add_limit_baseline(self, dataframe,):

        """ Adds a number of extrapolated data points to the dataframe 
            based on physical and chemical insights

        ARGS
        ----
        - dataframe (pd.DataFrame): the dataframe to which the data is added

        RETURNS
        -------
        - dataframe (pd.DataFrame): the updated dataframe

        """
    
        # define the inputs and the peak position
        inputs  = [[i/10, 1] for i in range(0, 10)]
        peak  = 515
        peaks = [peak for _ in range(len(inputs))]

        # list of unique molecules
        molecules = list(set(dataframe["molecule_name"]))

        # add new data to data objects
        for molecule in molecules:

            # get example data
            new_row = dataframe[dataframe["molecule_name"] == molecule].iloc[0]

            for i, input_ in enumerate(inputs):

                new_row["Cs_Pb_ratio"] = input_[1]
                new_row["AS_Pb_ratio"] = input_[0]
                new_row["Sample No."] = "baseline"
                new_row["baseline"] = True
                new_row["artificial"] = True
                new_row["monodispersity"] = 1
                new_row["peak_pos"] = peaks[i]
                new_row["poly_peak_pos"] = peaks[i]
                new_row["peak_pos_eV"] = nm_to_ev(peaks[i])
                new_row["fwhm"] = 0
                new_row["S/P"] = "SP"

                dataframe = dataframe._append(new_row, ignore_index = True)

        return dataframe


    def add_Toluene_baseline(self, dataframe,):

        """ Adds a baseline of the Toluene (no antisolvent) data to all antisolvents
        
        """

        print("Adding Toluene baseline")

        # list of unique molecules (exclude Toluene itself)
        molecules = list(set(dataframe["molecule_name"]))
        molecules.remove("Toluene")

        # get Toluene data (excluding the limit baselines)
        toluene_data = dataframe[dataframe["molecule_name"] == "Toluene"]
        toluene_data = toluene_data[dataframe["baseline"] == False]

        # iterate the rows of tolune data and add the baseline to the other molecules
        for index, row in toluene_data.iterrows():
            for molecule in molecules:
                new_row = row.copy()
                new_row["molecule_name"] = molecule
                new_row["baseline"] = True
                new_row["Sample No."] = "baseline"
                new_row["monodispersity"] = 1
                new_row["artificial"] = False
                new_row["fwhm"] = 0
                new_row["encoding"] = self.encode(molecule,)
                new_row["S/P"] = "SP"
                dataframe = dataframe._append(new_row, ignore_index = True)
            
        return dataframe



#### --------------------------------  DICIONARIES  ------------------------------- ####

    def get_dictionaries(self) -> dict:

        """ Read the molecule dictionary and ML dictionary and encoding dictionary from the json files
        """

        try:
            with open(self.molecule_dictionary_path, "r") as file:
                molecule_dictionary = json.load(file)
            
            with open(self.ml_dictionary_path, "r") as file:
                ml_dictionary = json.load(file)

            with open(self.encoding_path, "r") as file:
                encoding = json.load(file)

            return molecule_dictionary, ml_dictionary, encoding
        
        except FileNotFoundError:
            print("Could not find the molecule dictionary, encoding dictionary or the ML dictionary file")
            print("Please make sure the files are in the correct path and in .json format.")
            return None

