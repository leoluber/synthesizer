r""" Datastructure.py implements the Datastructure class used 
     throughout the project as main data structure  """
     # << github.com/leoluber >> 



from typing import Literal
import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from plotly import graph_objs as go
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# custom
from src.helpers import nm_to_ev, ev_to_nm




class Datastructure:

    """ Datastructure class

    General purpose data structure that reads in the synthesis data, spectral information 
    and the global attributes of the molecules, performs various selections and creates a 
    list of dict. objects that can be used for machine learning tasks.
    Can be easily extended to include more attributes or other target values.
    It is discouraged to use this class with different data sets!

    REQUIREMENTS
    ------------
    - all experimental data should be in "/data/raw/" and contain:
        - a folder "spectrum" with the spectral data in .txt or .csv files 
        - a .csv file with the synthesis data containing the sample numbers, molecule names, 
          synthesis parameters and other properties
        - a .csv file with the global attributes of the molecules (AntisolventProperties.csv)
        - a .json file with the molecule dictionary (molecule_dictionary.json)
        - a .json file with the molecule encoding (molecule_encoding.json)
        - a .json file with the ML dictionary (ml_dictionary.json)

    PARAMETERS
    ----------
    - synthesis_file_path:   path to the synthesis data file (csv)
    - spectral_file_path:    path to the spectral data files (txt, csv)
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
    >>> x, y = ds.get_training_data(["Cs_Pb_ratio", "AS_Pb_ratio"], "peak_pos")


    FEATURES
    ------------
    Can be found in the example data file

    """



    def __init__(
        self,
        synthesis_file_path:   str, 
        spectral_file_path:    str = "spectrum/",
        encoding:              Literal["one_hot", "geometry",] = "one_hot",
        wavelength_filter =    [400, 800],
        molecule =             "all",
        add_baseline =         False,
        monodispersity_only =  False,
        P_only =               False,
        ):
        

        # main stettings
        self.encoding          = encoding
        self.add_baseline      = add_baseline
        self.wavelength_filter = wavelength_filter


        # selection flags
        self.flags = {
                      "monodispersity_only": monodispersity_only,
                      "P_only"             : P_only,
                      "molecule"           : molecule
                      }
        
        # parameter names
        self.essential_parameters     = ["Sample No.", "c (PbBr2)", "c (Cs-OA)" , 
                                         "V (Cs-OA)","V (antisolvent)", "V (PbBr2 prec.)", 
                                         "antisolvent", ]
        self.non_essential_parameters = ["PLQY", "Reference", "PL_data", "Pb/I", "monodispersity",
                                         "Age of prec", "S/P", "suggestion", "NC shape", "polydispersity",
                                         "Centrifugation time [min]", "Centrifugation speed [rpm]"]
        self.parameters_to_normalize  = ["n_Cs", "n_Pb", "V_total", "V (Cs-OA)", "Pb/I", 
                                         "V (PbBr2 prec.)", "V (antisolvent)", "c (PbBr2)", 
                                         "c (Cs-OA)", "Centrifugation speed [rpm]","Centrifugation speed [rpm]"]
        

        # directories
        self.dataset =                  synthesis_file_path
        self.current_path =             os.getcwd()
        self.data_path_raw =            self.current_path  + "/data/raw/"
        self.data_path_processed =      self.current_path  + "/data/processed/"
        self.synthesis_file_path =      self.data_path_raw + synthesis_file_path
        self.processed_file_path =      self.data_path_processed + "processed_" + synthesis_file_path
        self.global_attributes_path =   self.data_path_raw + "AntisolventProperties.csv"
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
        
        self.densities = {f"{molecule}": self.global_attributes_df
                          .loc[self.global_attributes_df['antisolvent'] == molecule]['n [mol/L]']
                          .to_numpy()
                          .astype(float)[0] for molecule in self.molecule_names}
        




#### ----------------------------  READING & PREPROCESSING --------------------------- ####


    def read_synthesis_data(self) -> dict:

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
            df[key] = self.normalize(df[key], key)

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
        
        """ Reads the synthesis data from the csv file and returns a pandas dataframe
            with some basic error handling and parameter checking
        """

        # read the dataframe from the defined path
        if not os.path.exists(self.synthesis_file_path): 
            raise FileNotFoundError(f"File not found: {self.synthesis_file_path}")
        df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)  


        #check if the critical columns are present
        for key in self.essential_parameters:
            if key not in df.columns:
                raise KeyError(f"KeyError: {key} parameter not found in synthesis data")
            
        
        # print out unfamiliar columns
        print("Unfamiliar columns:")
        unfamiliar_columns = [key for key in df.columns 
                              if key not in self.essential_parameters + self.non_essential_parameters]
        print(unfamiliar_columns)

        # remove unknown molecules and replace "0" with "Toluene"
        recognized_molecules = list(self.molecule_dictionary.keys())
        df['antisolvent'] = df['antisolvent'].replace("0", "Tol")
        df = df[df['antisolvent'].isin(recognized_molecules)]

        return df


    def calculate_properties(self, df) -> pd.DataFrame:

        """ Calculate the essential properties from the data
            such as the Cs_Pb_ratio, AS_Pb_ratio, ...
            (if these raise an error, there is a fundamental issue with the provided dataset)
        """

        try:
            df["n_Cs"] = df["c (Cs-OA)"] * df["V (Cs-OA)"]
            df["n_Pb"] = df["c (PbBr2)"] * df["V (PbBr2 prec.)"]
            df['Cs_Pb_ratio'] = df["n_Cs"] / df["n_Pb"]
            df['AS_Pb_ratio'] = df["V (antisolvent)"] / df["n_Pb"]
            df["V (antisolvent)"] = df["V (antisolvent)"] + 0.000001
            df['V_total'] = df["V (Cs-OA)"] + df["V (antisolvent)"] + df["V (PbBr2 prec.)"]


        except KeyError:
            print("KeyError: central properties not found in synthesis data")
            return None

        return df
    

    def process_additional_properties(self, df) -> pd.DataFrame:

        """Handles specific properties of the synthesis data
            (especially the ones that are not found in all datasets)
            if these raise an error, there are workarounds in place to handle the issue
        """
    

        # (1) SYNTHESIS & CENTRIFUGATION TIME:  normalize the, log to handle the large range

        try:
            df["t_Rkt"] = self.normalize(np.log(df["t_Rkt"]), "t_Rkt")
            df["Centrifugation time [min]"] = self.normalize(df["Centrifugation time [min]"], "Centrifugation time [min]")
        except KeyError:
            print("KeyError: t_RKT or Centrifugation time [min], not found in synthesis data")
            df["t_Rkt"] = np.zeros(len(df["Sample No."]))
            df["Centrifugation time [min]"] = np.zeros(len(df["Sample No."]))


        # (2) PL_data: if not found, use sample number.txt instead

        if "PL_data" not in df.columns:
            print("KeyError: PL_data not found in synthesis data, using [sample_number].txt instead")
            df["PL_data"] = [str(x) + ".txt" for x in df["Sample No."]]


        # (3) MONODISPERSITY: can be a either a descrete or a continuous value 
        try:
            if set(df["monodispersity"]) <= {0, 1, 0.5, ""}: 
                #descrete values
                df["monodispersity"] = df["monodispersity"].replace("", 0)
                df["monodispersity"] = df["monodispersity"].astype(int)

            else:
                # continuous values
                df["monodispersity"] = self.normalize(df["monodispersity"], "monodispersity")

        except KeyError:
            print("KeyError: monodispersity not found in synthesis data")
            df["monodispersity"] = np.ones(len(df["Sample No."]))


        # (4) molecule_name: translate the molecule name to internal standard

        df["molecule_name"] = np.array([self.molecule_dictionary[molecule] for molecule in df["antisolvent"]])


        
        # (5) addditional columns: prepare the dataframe for later use

        # will be used later to mark the extrapolated data
        for key in ["baseline", "artificial", ]:
            df[key] = np.zeros(len(df["Sample No."]), dtype = bool)

        # encoding: one_hot or geometry encoding in list format
        df["encoding"] = [self.encode(molecule)
                        for molecule in df["molecule_name"]]



        # (7) Antisolvent correction: calculate the AS_Pb_ratio and n_As by considering the density and units
        df["n_As"] = np.zeros(len(df["Sample No."]))
        for index, row in df.iterrows():
            df["AS_Pb_ratio"][index] = row["AS_Pb_ratio"] * self.densities[row["molecule_name"]]
            df["n_As"][index] = row["V (antisolvent)"] * self.densities[row["molecule_name"]]

        # multiply AS_Pb_ratio by 10000
        df["AS_Pb_ratio"] = df["AS_Pb_ratio"] / 10000



        return df


    def add_spectral_data(self, df) -> pd.DataFrame:

        """ Read the spectral data from external files at /data/raw/ and
         add them to the dataframe """


        # define the columns for the spectral data
        for key in ["fwhm", "peak_pos", "poly_peak_pos", "peak_pos_eV"]:
            df[key] = np.zeros(len(df["Sample No."]))

        # iterate the rows and read the spectral data
        for index, row in df.iterrows():
            try:
                fwhm, peak_pos, poly_peak_pos, peak_pos_eV = self.read_spectrum(self.spectrum_path + row["PL_data"])
                df["fwhm"][index] = fwhm
                df["peak_pos"][index] = peak_pos
                df["poly_peak_pos"][index] = poly_peak_pos
                df["peak_pos_eV"][index] = peak_pos_eV

            except TypeError:
                print(f"TypeError: {row['PL_data']} not found in spectral data")
                continue


        # remove all rows with missing spectral data
        df = df[df["peak_pos"] != 0]


        # classify the product
        df["NPL_type"] = [self.get_NPL_type(peak_pos) for peak_pos in df["peak_pos"]]


        return df



#### ------------------------------  TRAINING DATA  ------------------------------- ####


    def get_training_data(self, training_selection: list, target: str, encoding = False )-> tuple:

        """ reads processed_data.csv and returns the training data as a numpy arrays 
            
            USAGE: after the data is read in and processed through read_synthesis_data()
            the training data can be generated using this function
        """

        # read the processed data
        data_frame = pd.read_csv(self.processed_file_path, delimiter= ";", header= 0)


        #### --------------- TEMPORARY CHANGES --------------- ####
        data_frame = data_frame[data_frame["Sample No."].astype(int) >= 120]
        data_frame  = data_frame[data_frame["Pb/I"].astype(float) >0.4]


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
            data_frame = data_frame[data_frame['monodispersity'].astype(int) == 1]

        if self.flags["P_only"]:
            data_frame = data_frame[data_frame['S/P'] != "S"]

        if target == "PLQY":
            data_frame = data_frame[data_frame['PLQY'] != 0]

        # get the training data
        x = data_frame[training_selection].to_numpy()
        y = data_frame[target].to_numpy()

        # add the encoding if requested
        if encoding:
            encodings = np.array([json.loads(x) for x in data_frame["encoding"]])
            x = np.concatenate((encodings, x), axis = 1)


        return x, y, data_frame
            

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

        if not os.path.exists(path):
            return 0,0,0,0
    
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
        peak_pos_eV = peak_pos
        poly_peak_pos = np.average(np.array(energies), weights = spectrum)


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
        poly_peak_pos = ev_to_nm(poly_peak_pos)
        wavelengths = [ev_to_nm(x) for x in energies]

        if peak_pos < 400:
            print(f"Peak position below 400 nm: {peak_pos} nm")
            return None
    

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

        for key, value in self.ml_dictionary.items():
            if value[0] <= peak_pos <= value[1]:
                return float(key)
            
        return 0.




#### ----------------------------  BASELINE ADDITION  --------------------------- ####

    def add_limit_baseline(self, dataframe,):

        """ Adds a number of extrapolated data points to the dataframe 
            based on physical and chemical insights
        """
    
        # define the inputs and the peak position
        inputs  = [[i/10, 1] for i in range(0, 10)]
        peak  = 515

        peaks = [peak for _ in range(len(inputs))]

        # list of unique molecules
        molecules = list(set(dataframe["molecule_name"]))

        # add new data to data objects
        print(inputs)
        for molecule in molecules:
            for i, input_ in enumerate(inputs):
                new_row = {}
                new_row["Cs_Pb_ratio"] = input_[1]
                new_row["AS_Pb_ratio"] = input_[0]
                new_row["Sample No."] = "baseline"
                new_row["molecule_name"] = molecule
                new_row["baseline"] = True
                new_row["artificial"] = True
                new_row["encoding"] = self.encode(molecule,)
                new_row["peak_pos"] = peaks[i]
                new_row["poly_peak_pos"] = peaks[i]
                new_row["peak_pos_eV"] = nm_to_ev(peaks[i])
                new_row["fwhm"] = 0

                dataframe = dataframe._append(new_row, ignore_index = True)
        
        print(dataframe[['Sample No.', 'molecule_name', 'Cs_Pb_ratio', 'AS_Pb_ratio', 'peak_pos']])

        return dataframe


    def add_Toluene_baseline(self, dataframe,):

        """ Adds a baseline of the Toluene (no antisolvent) data to all antisolvents"""
        print("Adding Toluene baseline")

        # list of unique molecules
        molecules = list(set(dataframe["molecule_name"]))

        # we don't want to add the Toluene data to the Toluene data
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
                new_row["artificial"] = False
                new_row["fwhm"] = 0
                new_row["encoding"] = self.encode(molecule,)
                dataframe = dataframe._append(new_row, ignore_index = True)
            
        return dataframe




#### --------------------------------  DICIONARIES  ------------------------------- ####

    def get_dictionaries(self) -> dict:
        
        """ 
            Read the molecule dictionary and ML dictionary from the json files
        """

        try:
            with open(self.molecule_dictionary_path, "r") as file:
                molecule_dictionary = json.load(file)
            
            with open(self.ml_dictionary_path, "r") as file:
                ml_dictionary = json.load(file)

            with open(self.geometry_path, "r") as file:
                encoding = json.load(file)

            return molecule_dictionary, ml_dictionary, encoding
        
        except FileNotFoundError:
            print("Could not find the molecule dictionary, encoding dictionary or the ML dictionary file")
            print("Please make sure the files are in the correct path and in .json format.")
            return None
