import numpy as np
from torch_geometric.data import Data   # bit of a relict from the GNN implementation, but still useful
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helpers import *


"""
    General porpouse data structure that reads in the synthesis data and the global attributes 
    of the molecules and creates a list of torch_geometric Data objects; can be easily adjusted 
    to include more global attributes or other target values.
    DATA DIRECTORY should be at "/data/" relative to the current working directory and contain:
    - a folder "spectrum" with the spectral data in .txt files (csv)
    - .json files for the molecules with the atoms, coordinates and bonds (PubChem-formatted)
    - AntisolventProperties.csv with the global attributes of the molecules
    - a .csv file with the synthesis data (synthesis_file_path) containing the sample numbers, molecule names 
    and synthesis parameters
"""



class Datastructure:
    def __init__(self,
                 synthesis_file_path,               # path to the synthesis data file
                 target = "FWHM",                   # Options: MONODISPERSITY, PEAK_POS, PLQY, FWHM, NPL_TYPE
                 output_format = "TENSOR",          # LIST or TENSOR (for final output format of the Data objects)
                 wavelength_filter = [1, 1000],     # filter for the spectrum, always in "NM"
                 exclude_no_star = False,           # exclude samples with bad PLQY
                 exclude_PLQY = False,
                 wavelength_unit = "EV",            # "EV" or "NM
                 molecule = "all",                  # "all" or specific molecule
                 normalization = True,              # normalize the data
                 monodispersity_only = False,       # only include samples with monodispersity
                 PLQY_threshold = 0.0,              # exclude samples with PLQY below threshold
                 ):
        

        self.target = target


        # flags
        self.exclude_no_star = exclude_no_star
        self.exclude_PLQY = exclude_PLQY
        self.wavelength_unit = wavelength_unit
        self.molecule = molecule
        self.PLQY_threshold = PLQY_threshold
        self.monodispersity_only = monodispersity_only

        # directories
        self.current_path =  os.getcwd()
        self.data_path = self.current_path + "/data/"
        self.synthesis_file_path = self.data_path + synthesis_file_path
        self.global_attributes_path = self.data_path + "AntisolventProperties.csv"
        self.spectrum_path = self.data_path + "spectrum/"

        # central data list that contains the Data objects
        self.max_min = {}                                               # dictionary to store the max and min values of the parameters
        self.data = []                                                  # should not be accessed, use get_data() instead
        if wavelength_unit == "EV":
            self.wavelength_filter = [nm_to_ev(wavelength_filter[0]), nm_to_ev(wavelength_filter[1])]
        else:
            self.wavelength_filter = wavelength_filter
        self.normalization = normalization
        
        # reading data from files, selections adjusted here
        self.global_attribute_selection =       [] #"dielectric constant (-)",  "Hansen parameter hydrogen bonding (MPa)1/2", "dipole moment (D)", "Gutman donor number (kcal/mol)", "viscosity (mPa ?s)" ,]
        self.synthesis_training_selection =     [ "AS_Pb_ratio", "V (antisolvent)",  "V (Cs-OA)", "V (PbBr2 prec.)", "c (PbBr2)",  "c (Cs-OA)", "Pb_Cs_ratio", "c (OlAm)", "c (OA)"] #

        self.synthesis_selection =              [ "c (PbBr2)", "c (OlAm)", "c (OA)", "V (Cs-OA)", "c (Cs-OA)" ,"V (antisolvent)", "V (PbBr2 prec.)"]
        
        self.total_training_parameter_selection = self.synthesis_training_selection + self.global_attribute_selection
        #self.total_training_parameter_selection.append("polarity")

        self.synthesis_data = self.read_synthesis_data()                                # container of synthesis data arrays,           read in during initialization
        self.global_attributes_df = self.read_global_attributes_and_nomalize()          # dataframe of global attributes,               read in during initialization

        # dictionaries (might need to be adjusted for new molecules)
        self.molecule_dictionary, self.atom_to_num, self.num_to_atom, self.ml_dictionary = self.get_molecule_dictionary()

        # general
        self.target = target
        self.output_format = output_format

        # list of molecule names, don't ask ...
        self.molecule_names = ["Toluene", "EthylAcetate", "MethylAcetate", "Acetone", "Ethanol", "Methanol", "Isopropanol", "Butanol", "Tert-Butanol", "Propanol", "Acetonitrile", "Dimethylformamide", "Dimethylsulfoxide", "Butanone", "Cyclopentanone"]



    def get_data(self):
        
        """
            - loop that iterates through the samples, reads in all the extra information and represents it as a Data object
            - synthesis data and the gloal attributes dataframe are read in during initialization 
            (accessed via self.synthesis_data["property_name"][index])
            - molecule specific data is read in during the loop
        """

        for index, sample_number in enumerate(self.synthesis_data["sample_numbers"]):
           
            # translates the data dependent molecule name to inherent module code (because I'm not a chemist and this is a mess)
            molecule_name = self.molecule_dictionary[self.synthesis_data["molecule_names"][index]]
           
            # get the spectrum data, that matches "sample_number".txt and read in the peak position and FWHM
            path = self.spectrum_path + str(sample_number) + ".txt"
            if not os.path.exists(path): 
                continue
            fwhm, peak_pos, spectrum, wavelength = self.read_spectrum(sample_number)
            spectrum = compress_spectrum(spectrum=spectrum[0:150])                          #(if the spectrum is used as input, it should be compressed)


            # flags
            if self.molecule != "all" and self.molecule != molecule_name:
                continue  
            if peak_pos <= min(self.wavelength_filter) or peak_pos >= max(self.wavelength_filter):
                continue
            if self.target == "PLQY" and self.synthesis_data['PLQY'][index] == 0:
                continue
            if self.target == "PLQY" and self.synthesis_data['PLQY'][index] < self.PLQY_threshold:
                continue
            if self.exclude_no_star and self.synthesis_data['include PLQY'][index] != 1 :
                continue
            if self.monodispersity_only and self.synthesis_data['monodispersity'][index] != "1":
                continue


            # classify the NPL type
            NPL_type = self.get_NPL_type(peak_pos)


            # get global attributes df for the molecule
            global_attributes_df = self.global_attributes_df.loc[self.global_attributes_df['antisolvent'] == molecule_name]
            
            
            # calculate extra attributes (relative polarity is not normalized yet for this reason)
            RelPolarity = global_attributes_df['relative polarity (-)'].to_numpy().astype(float)[0]
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
            #synthesis_parameters.append(polarity)        #---------------------->  add the extra attributes here
            total_parameters = synthesis_parameters + global_attributes


            # one hot encoding for the molecule
            one_hot_molecule = [0] * len(self.molecule_names)
            one_hot_molecule[self.molecule_names.index(molecule_name)] = 1
            


            # set the target
            #if self.target   ==     "MONODISPERSITY": target = self.synthesis_data["monodispersity"][index]
            if self.target   ==     "FWHM": target = fwhm
            elif self.target ==     "PEAK_POS": target = peak_pos
            elif self.target ==     "NPL_TYPE": target = NPL_type
            elif self.target ==     "PLQY": target = self.synthesis_data["PLQY"][index]
            elif self.target ==     "SYNTH": target = self.synthesis_data["Pb_Cs_ratio"][index]     # for reverse approach
            else:                   print("No target specified")
            
        
            ### ----------------------  DATA OBJECT CREATION  ---------------------- ###

            # if speed is a concern, drop the unnecessary attributes
            data_point = Data(
                            y = target,
                            spectrum = spectrum,
                            peak_pos = peak_pos,
                            NPL_type = NPL_type,
                            fwhm = fwhm,
                            sample_number = sample_number,
                            total_parameters = total_parameters,
                            global_attributes = global_attributes,
                            synthesis_parameters = synthesis_parameters,
                            parameter_selection = self.total_training_parameter_selection,
                            one_hot_molecule = one_hot_molecule,
                            molecule_name = molecule_name,
                            )
            
            self.data.append(data_point)
        return self.data
    


#### ------------------------------------------  helper functions  --------------------------------------------- ####

    # read in spectrum of a single sample, returns FWHM (only for integer time steps)
    def read_spectrum(self, sample_number):
        
        #path
        path = self.spectrum_path + str(sample_number) + ".txt"
    
        wavelength, spectrum = [], []
        with open(path, "r") as filestream:
            for line in filestream:
                if line == '': break
                x, A, y  = line.split(",")    # of the form (wavelength in nm, amplitude, norm. amplitude)
                if self.wavelength_unit == "EV":
                    wavelength.append(nm_to_ev(float(x)))           # ] use nm_to_ev for eV              
                else:
                    wavelength.append(float(x))                     # ] or use nm
                spectrum.append(float(y))

        # get peak position
        max_index = spectrum.index(max(spectrum))
        peak_pos = wavelength[max_index]

        #logic for FWHM, could be changed
        half_max = 0.5
        left_index, right_index = max_index, max_index
        while spectrum[left_index] > half_max:
            left_index -= 1
        while spectrum[right_index] > half_max:
            right_index += 1
        fwhm = abs(wavelength[left_index] - wavelength[right_index])
        return fwhm, peak_pos, spectrum, wavelength

    # norm. dataframes or arrays (use with rows or columns); includes a check for zero range
    def normalize(self, a, name):
        self.max_min[name] = [a.max(), a.min()]   # store the max and min values for denormalization
        if (a.max() - a.min()) == 0: 
            print(f"Normalization error: {a}  has no range; returning original array")
            return a
        #return (a-a.min())/(a.max()-a.min())
        return a / a.max() 
    
    # denormalize a specific parameter
    def denormalize(self, a, name):
        max_val, min_val = self.max_min[name]
        #return a * (max_val - min_val) + min_val
        return a * max_val

    # normalize targets
    def normalize_target(self, data_objects):
        target_name = self.target
        targets = [data.y for data in data_objects]

        max_val, min_val = max(targets), min(targets)
        if (max_val - min_val) == 0: 
            print(f"Normalization error: {targets}  has no range; returning original array")
            return data_objects
        for data in data_objects:
            data.y = (data.y - min_val) / (max_val - min_val)
        
        self.max_min[target_name] = [max_val, min_val]
        return data_objects

    # classify the NPL type from the peak position
    def get_NPL_type(self, peak_pos):
        if peak_pos is None: 
            print("No peak position found")
            return None
        for key, value in self.ml_dictionary.items():
            if self.wavelength_unit == "EV":
                peak_pos = ev_to_nm(peak_pos)
            if value[0] <= peak_pos <= value[1]:
                return float(key)
        return 0



#### ------------------------------------------  init. helpers  --------------------------------------------- ####

    # read in the global graph attributes from AntisolventProperties.csv for a certain molecule
    def read_global_attributes_and_nomalize(self):
        
        # path
        df = pd.read_csv( self.global_attributes_path, delimiter= ';', header= 0)
        
        # normalize the columns
        if self.normalization:
            for column in self.global_attribute_selection:
                df[column] = self.normalize(df[column], column)
        return df
    
    
    # read in the synthesis data from dataframe and return as container  ----> could be changed to keep pandas dataframe for initialization
    def read_synthesis_data(self):
    
        # read in the synthesis data
        df = pd.read_csv(self.synthesis_file_path, delimiter= ';', header= 0)                                                                 

        # normalize the columns
        if self.normalization:
            for key in self.synthesis_selection:
                df[key] = self.normalize(df[key], key)

        # calculate the Pb/Cs ratio
        Pb_Cs_ratio = (df["c (PbBr2)"] * df["V (PbBr2 prec.)"] ) / (df["c (Cs-OA)"] * df["V (Cs-OA)"] )
        AS_Pb_ratio = df["V (antisolvent)"] / (df["V (PbBr2 prec.)"] * df["c (PbBr2)"] )

        if self.normalization:
            #Pb_Cs_ratio = self.normalize(Pb_Cs_ratio, "Pb_Cs_ratio")
            AS_Pb_ratio = self.normalize(AS_Pb_ratio, "AS_Pb_ratio")                                            # temporary normalization trick, otherwise the synthesizer doesn't work

        synthesis_data = {key  : df[key].to_numpy().astype(float) for key in self.synthesis_selection}          # to normalize or not to normalize the data, that is the question...


        # add ratio (probably the most expressive parameter)
        synthesis_data["Pb_Cs_ratio"] = Pb_Cs_ratio.to_numpy().astype(float)
        synthesis_data["AS_Pb_ratio"] = AS_Pb_ratio.to_numpy().astype(float)

        # add target related data
        synthesis_data["monodispersity"] = df["monodispersity"].to_numpy()     #.astype(float)
        synthesis_data["sample_numbers"] = df['Sample No.'].to_numpy().astype(int)
        synthesis_data["PLQY"] =           df["PLQY"].to_numpy().astype(float)
        synthesis_data["include PLQY"] =   df["include PLQY"].to_numpy()

        # add the sample numbers and molecule names (not normalized obviously)
        synthesis_data["sample_numbers"] = df['Sample No.'].to_numpy().astype(int)
        synthesis_data["molecule_names"] = df['antisolvent'].to_numpy()

        
        return synthesis_data



#### ------------------------------------------  PLOTTING  --------------------------------------------- ####

    # plotting the Data in Parameter Space
    def plot_data(self, var1, var2, var3, parameters):
        """
            Scatter plot of the data in parameter space, for visualization purposes
        """
        index1 = parameters.index(var1)
        index2 = parameters.index(var2)
        index3 = parameters.index(var3)

        one_hot = [data.one_hot_molecule for data in self.data]
        molecule_index = [np.argmax(one_hot) for one_hot in one_hot]

        x = [data.total_parameters[index1] for data in self.data]
        y = [data.total_parameters[index2] for data in self.data]
        z = [data.total_parameters[index3] for data in self.data]
        c = [data.y for data in self.data]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=c, cmap='viridis', linewidth=5, alpha=1)

        # Add colorbar
        cbar = fig.colorbar(ax.scatter(x, y, z, c=c, cmap='viridis', linewidth=5, alpha=1))
        cbar.set_label(self.target)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel(var3)
        #plt.show()
        return fig, ax

    # plotting the peak positions for As/Pb ratio for a specific molecule
    def plot_As_Pb_peak_pos(self, molecule_name):
        """
            Plots the peak position over AS/Pb ratio
        """
        AS_Pb = [data.total_parameters[0] for data in self.data if data.molecule_name == molecule_name]
        peak_pos = [data.peak_pos for data in self.data if data.molecule_name == molecule_name]

        # fitting a Sigmoid function
        Sigmoid = lambda x, a, b, c, d: (a / (1 + np.exp(-b * (x - c)))) + d
        bounds = ([50, 0, 0, 460], [65, 0.5, 300, 463])
        popt, pcov = curve_fit(Sigmoid, AS_Pb, peak_pos, bounds=bounds)
        print(f"Optimal parameters: {popt}")

        #write to csv
        #df = pd.DataFrame({"AS_Pb_ratio": AS_Pb, "Peak Position": peak_pos})
        #df.to_csv(f"{molecule_name}_AS_peak_pos.csv", index=False)

        #plotting
        fig, ax = plt.subplots()
        x_vec = np.linspace(min(AS_Pb), max(AS_Pb), 500)
        label = f"y = $\\frac{{{round(popt[0])}}}{{1 + e^{{-{round(popt[1],2)}(x - {round(popt[2])})}}}} + {round(popt[3])}$"
        ax.plot(x_vec, Sigmoid(x_vec, *popt), color="red", label = label)
        ax.scatter(AS_Pb, peak_pos)
        ax.set_xlabel("AS/Pb ratio")
        ax.set_ylabel("Peak Position")
        ax.set_title(f"Peak Position vs AS/Pb ratio for {molecule_name}")
        ax.legend(fontsize = 15)

        # save the plot
        plt.savefig(f"{molecule_name}_AS_peak_pos_Sigmoid.png")

        #plt.show()
        return fig, ax

    # plotting the average target value for each molecule and NPL type
    def plot_avg_target(self):
        """
            Plots the average target value for each molecule and NPL type
        """

        map = np.zeros((len(self.molecule_names), 9))
        stds = map.copy()
        for molecule in self.molecule_names:
            for NPL_type in range (1, 10):
                targets = [data.y for data in self.data if data.molecule_name == molecule and data.NPL_type == NPL_type]
                count = len(targets)
                if count == 0:
                    continue
                avg_target = np.mean(targets)
                std_target = np.std(targets)
                stds[self.molecule_names.index(molecule), NPL_type - 1] = std_target
                map[self.molecule_names.index(molecule), NPL_type - 1] = avg_target
        
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

    # dictionaries to standardize names inside this module (might be adjusted)
    def get_molecule_dictionary(self):

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
        
        atom_to_num = {'H' : 1,
              'C' : 6,
              'O' : 8,
              'N' : 7,
              'S' : 16
              }
        
        num_to_atom = {v: k for k, v in atom_to_num.items()}   #inverse dictionary     number --> atom

        return molecule_dictionary, atom_to_num, num_to_atom, ml_dictionary
