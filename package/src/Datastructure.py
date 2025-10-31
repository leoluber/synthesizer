
""" 
    Module:         Datastructure.py
    Project:        Synthesizer: Chemistry-Aware Machine Learning for 
                    Precision Control of Nanocrystal Growth 
                    (Henke et al., Advanced Materials 2025)
    Description:    Class for handling the data structure and preprocessing
                    in the context of the Synthesizer project
    Author:         << github.com/leoluber >> 
    License:        MIT
    Year:           2025
"""


# ----------------- #
import numpy as np
import os
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# ----------------- #



class Datastructure:


    """ Datastructure class for the Synthesizer project

    General purpose data structure that reads in the synthesis data, spectral information 
    and the global attributes of the molecules. It performs various selections and saves 
    the data in a standardized .csv file that can be used for machine learning tasks.
    Can be easily extended to include more attributes or other target values.


    DATA REQUIREMENTS
    -----------------
    - all experimental data should be in "/data/raw/" and contain:
        - a .csv file with the synthesis data containing the sample numbers, molecule names, 
          synthesis parameters and other properties as well as the results of optical 
          characterization (dataset_synthesizer.csv)
        - a .csv file with the global attributes of the molecules (AntisolventProperties.csv)
        - a .json file with the molecule dictionary (molecule_dictionary.json)
        - a .json file with the molecule encodings (molecule_encoding.json)

    [blueprints for the .json files are provided in the repository, adjust as you see fit
    NOTE: how to do this properly is discussed in self.get_dictionaries()]

        
    ARGS
    ----
    - synthesis_file_path (str):   path to the synthesis data file (.csv)
    - wavelength_filter (tuple):   filter for the peak position (list of two values in nm)
    - molecule (str):              molecule to be selected ("all" or specific molecule)
    - add_baseline (bool):         add baseline data to the data objects (see publication for details)
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
        wavelength_filter =    [400, 600], #nm
        molecule =             "all",
        encoding =             "geometry",
        add_baseline =         False,
        monodispersity_only =  False,
        P_only =               False,
        S_only =               False,
        ):
        
        # check S/P
        if S_only and P_only:
            raise ValueError("S_only and P_only cannot be True at the same time")

        # main settings
        self.add_baseline      = add_baseline
        self.wavelength_filter = wavelength_filter
        self.encoding          = encoding

        # selection flags (changes here need to be reflected in "self.get_training_data()")
        self.flags = {"monodispersity_only": monodispersity_only,
                      "P_only"             : P_only,
                      "S_only"             : S_only,
                      "molecule"           : molecule}
        
        
        ##### ------------ TODO: IMPLEMENT YOUR CUSTOM FEATURE NAMES HERE ----------- ####
        # feature names (essential and non-essential)
        self.essential_parameters     = ["Sample No.", "c (PbBr2)", "c (Cs-OA)" , 
                                         "V (Cs-OA)","V (antisolvent)", "V (PbBr2 prec.)", 
                                         "antisolvent", "S/P", "monodispersity",]
        
        # parameters used for training should be normalized/rescaled to a common range 
        # however, ratios should not be normalized (further discussed in publication)
        self.parameters_to_normalize  = ["V_total", "V (Cs-OA)", "V (PbBr2 prec.)", 
                                         "V (antisolvent)", "c (PbBr2)", "c (Cs-OA)",]
        ##### ----------------------------------------------------------------------- ####


        # directories
        self.dataset =                  synthesis_file_path
        self.current_path =             os.getcwd()   # needs to be changed when using linux
        self.data_path_raw =            self.current_path  + "/data/raw/"
        self.synthesis_file_path =      self.data_path_raw + synthesis_file_path
        self.processed_file_path =      self.current_path  + "/data/processed/processed_" + synthesis_file_path
        self.global_attributes_path =   self.data_path_raw + "AntisolventProperties.csv"
        self.encoding_path=             self.data_path_raw + "molecule_encoding.json"
        self.molecule_dictionary_path = self.data_path_raw + "molecule_dictionary.json"


        # dictionaries for molecule names, geometries and atom numbers
        self.molecule_dictionary, self.encoding_dictionary \
            =  self.get_dictionaries()

        # list of all relevant molecule names
        self.molecule_names = list(self.molecule_dictionary.values())

        # for normalization and denormalization
        self.max_min = {}

        # molecule attributes
        self.global_attributes_df =  pd.read_csv(self.global_attributes_path, 
                                                 delimiter= ';', header= 0)
        
        # read the antisolvent concentrations
        self.concentrations = {f"{molecule}": self.global_attributes_df
                          .loc[self.global_attributes_df['antisolvent'] == molecule]['c [mol/L]']
                          .to_numpy()
                          .astype(float)[0] for molecule in self.molecule_names}
        


# ------------------------------------------------------------------
#                       READING & PREPROCESSING
# ------------------------------------------------------------------


    def read_synthesis_data(self):

        """ Read in the synthesis data from dataframe and write to processed_file_path
        
        The synthesis data is read in from the synthesis file and normalized.
        The data is then stored in a standardized .csv file that can be used for
        machine learning tasks.
        - NOTE: baseline data can be added based on physical and chemical insights
                (further discussed in publication)
        --> ratios are NOT normalized, instead units are adjusted to create a range of 0-1

        """

        print("Reading synthesis data...")

        # get initial dataframe
        df = self.get_synthesis_dataframe()

        # normalize relevant properties
        for key in self.parameters_to_normalize:
            try:
                df[key] = self.normalize(df[key], key)
            except KeyError:
                print(f"KeyError: {key} not found in synthesis data")
                continue
        
        # add baseline data
        if self.add_baseline:
            for key in ["baseline", "artificial", ]:
                df[key] = np.zeros(len(df["Sample No."]), dtype = bool)

            df = self.add_limit_baseline(df)
            df = self.add_Toluene_baseline(df)

        # write updated dataframe to csv
        df.to_csv(self.processed_file_path, sep = ";")

        print("Data read and processed successfully!")
        return 0
    

    def get_synthesis_dataframe(self) -> pd.DataFrame:
        
        """ Reads the synthesis data from the csv file 

        RETURNS
        -------
        - df (pd.DataFrame): the synthesis data

        RAISES
        ------
        - FileNotFoundError: if the synthesis file is not found
        - KeyError: if essential parameters are missing in the synthesis data

        """

        # read the dataframe from the defined path
        if not os.path.exists(self.synthesis_file_path):
            raise FileNotFoundError(f"File not found: {self.synthesis_file_path}")
        df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)  

        #check if the critical columns are present
        for key in self.essential_parameters:
            if key not in df.columns:
                raise KeyError(f"KeyError: {key} parameter not found in synthesis data")
        return df



# ------------------------------------------------------------------
#                          SELECT TRAINING DATA 
# ------------------------------------------------------------------ 


    def get_training_data(self, training_selection: list, 
                          target: str, encoding = False, remove_baseline = False )-> tuple:

        """ Reads processed_data.csv and returns the training data as a numpy arrays 
            
        USAGE
        -----
        after the data is read in and processed through read_synthesis_data()
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

        # remove the baseline data if requested (its added as default)
        if remove_baseline:
            data_frame = data_frame[data_frame["baseline"] == False]

        # check if all relevant keys are in the dataframe
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
            data_frame = data_frame[data_frame['monodispersity'] != 0]

        if self.flags["P_only"]:
            print("Removing non-P type samples")
            data_frame = data_frame[data_frame['S/P'] != "S"]
        
        elif self.flags["S_only"]:
            print("Removing non-S type samples")
            data_frame = data_frame[data_frame['S/P'] != "P"]

        # remove samples with target = 0 or NaN
        data_frame = data_frame[data_frame[target] != 0]
        data_frame = data_frame.dropna(subset = [target])

        # get the training data
        x = data_frame[training_selection].to_numpy()
        y = data_frame[target].to_numpy()

        # add the encoding if requested
        if encoding:
            encodings = np.array([json.loads(x) for x in data_frame["encoding"]])
            x = np.concatenate((encodings, x), axis = 1)

        return x, y, data_frame



# ------------------------------------------------------------------
#                                HELPERS 
# ------------------------------------------------------------------ 

    def normalize(self, a, name):

        """ Norm. dataframes or arrays and stores the max and min values for denormalization 
        """
        self.max_min[name] = [a.max(), a.min()]

        if a.max() == a.min(): 
            print(f"Normalization issue: {a}  has no range; returning original array")
            return a
        return (a-a.min())/(a.max()-a.min())
    

    def denormalize(self, a, name):

        """ Denormalizes a value based on the max and min values 
        """
        try:
            max_val, min_val = self.max_min[name]
        except KeyError:
            print(f"KeyError: {name} not found in max_min dictionary")

        return a * (max_val - min_val) + min_val


    def encode(self, molecule_name, enc = None) -> list:

        """ Encodes the molecule name to a geometry encoding 
        
        RETURNS
        -------
        - geo_encoding (list): the geometry encoding of the molecule """


        if enc is None:
            enc = self.encoding

        match enc:
            case "geometry":
                return [float(x) for x in self.encoding_dictionary[molecule_name]]
            
            ### -- ADD YOUR OWN ENCODING SCHEMES HERE -- ###

            case _:
                print("Invalid encoding type")




# ------------------------------------------------------------------
#                            BASELINE ADDITION 
# ------------------------------------------------------------------ 

    def add_limit_baseline(self, dataframe,):

        """ Adds a number of extrapolated data points to the dataframe 
            based on physical and chemical insights

            --> only used for PL peak position prediction
        """

        print("Adding limit baselines...")
    
        # define the artificial inputs and the peak position
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
                new_row["fwhm"] = 0
                new_row["S/P"] = "SP"

                dataframe = dataframe._append(new_row, ignore_index = True)

        return dataframe


    def add_Toluene_baseline(self, dataframe,):

        """ Adds a baseline of the Toluene (no antisolvent) data to all antisolvents
        
        """

        print("Adding Toluene baseline...")

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



# ------------------------------------------------------------------
#                                DICTIONARIES
# ------------------------------------------------------------------

    def get_dictionaries(self) -> dict:

        """ Read the molecule dictionary and encoding dictionary from the json files

        molecule_dictionary: maps molecule names in the csv file to
            standardized molecule names in the synthesizer framework
            (e.g. "MeOH" --> "Methanol")
            NOTE: if a different naming scheme is used in the synthesis data,
            adjust the molecule_dictionary.json file in the /data/raw/ folder

        encoding_dictionary: maps molecule names to geometry encodings
            NOTE: the simplest way to implement an encoding is to change 
            the molecule_encoding.json file in the /data/raw/ folder

        """

        try:
            with open(self.molecule_dictionary_path, "r") as file:
                molecule_dictionary = json.load(file)

            with open(self.encoding_path, "r") as file:
                encoding = json.load(file)

            return molecule_dictionary, encoding
        
        except FileNotFoundError:
            print("Could not find the molecule dictionary or the encoding dictionary file")
            print("Please make sure the files are in the correct " \
            "path and in .json format.")
            return None


