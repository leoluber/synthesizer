"""Data Preprocessor to perform data selection and add residual targets (compatible with the Datastructure class)"""
   # < github.com/leoluber > 


import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from typing import Literal

#custom
from GaussianProcess import GaussianProcess




class Preprocessor:


    """ Data Preprocessor

    This class is used to add constraints and perform data selection for the Datastructure class
    to retain a more robust dataset for the regression models
    - constraints can be chosen from a range of approaches
    - the target values can be converted to residual values to improve the regression models
      (Details in the add_residual_targets() method)
    - the target values can be converted to residual values based on the average target value
      (Details in the add_residual_targets_avg() method)

      
    PARAMETERS
    ----------
    selection_method: selection method (str)
        - "FWHM": remove data points with "relatively" high FWHM values
        - "PEAK_SHAPE": remove data points with poor peak shape
        - "CS_PB": remove data points with high Cs/Pb ratio
        - "hard_FWHM": apply hard NPL specific constraints on the FWHM values
    
    fwhm_margin: margin for the FWHM selection (float)
    peak_error_threshold: threshold for the peak shape selection (float)
    mode: mode: "NM" or "EV" (str)

    USAGE
    -----
    >>> ds = Preprocessor(selection_method= ["PEAK_SHAPE",], fwhm_margin=-0.01, peak_error_threshold=0.0002)
    >>> data_objects = ds.select_data(data_objects)
    >>> data_objects = ds.add_residual_targets(data_objects)
    >>> #data_objects = ds.add_residual_targets_avg(data_objects

    """


    def __init__(self,
                selection_method: Literal["DIST_CORR", "FWHM", "PEAK_SHAPE", "CS_PB", "hard_FWHM"] = ["PEAK_SHAPE"],
                fwhm_margin: float = 3,
                peak_error_threshold: float = 0.00015,
                mode : Literal["NM", "EV"] = "NM"
                ):


        # settings
        self.mode = mode

        # main attributes
        self.selection_method = selection_method
        self.fwhm_margin = fwhm_margin
        self.peak_error_threshold = peak_error_threshold




#### ---------------------------- SELECTION ---------------------------- ####

    def select_data(self, data_objects: list)-> list:
        """
            Select data points according to the selection method
            -> all selection methods specified in the selection_method list are applied
        """

        selection = data_objects

        if "FWHM"        in self.selection_method:
            selection =  self.fit_fwhm_trendline(data_objects=selection)

        if "PEAK_SHAPE"  in self.selection_method:
            selection =  self.peak_selection(data_objects=selection)

        if "CS_PB"       in self.selection_method:
            selection =  self.set_Cs_Pb_limit(data_objects=selection)

        if "hard_FWHM"   in self.selection_method:
            selection =  self.hard_FWHM_constraints(data_objects=selection)

        if selection is None:
            raise ValueError("Selection method not recognized")
        
        return selection



    def fit_fwhm_trendline(self, data_objects: list)-> list:
        """
            Fit a trendline to the FWHM values
            -> removes data points with FWHM values above the trendline + margin
        """
        
        # split the data into baseline and current data (data with and without FWHM values)
        current_data   = [data for data in data_objects if "fwhm" in data]
        baseline_data  = [data for data in data_objects if "fwhm" not in data]
        fwhm_values    = [data["fwhm"] for data in current_data]
        peak_positions = [[data["peak_pos"]] for data in current_data]


        # fit a trendline (KRR) to the FWHM values
        krr = KernelRidge(kernel="rbf", alpha=0.001, gamma=0.001)
        krr.fit(peak_positions, fwhm_values)


        # visualize the trendline
        x_vec       = np.linspace(min(peak_positions), max(peak_positions), 1000)
        y           = [krr.predict([x]) for x in x_vec]
        y_margin    = [krr.predict([x]) + self.fwhm_margin for x in x_vec]
        

        plt.plot(x_vec, y, c="red")
        plt.plot(x_vec, y_margin, c="red", linestyle="--")
        plt.scatter(peak_positions, fwhm_values, c="blue")
        plt.show()
    

        # if the FWHM values are above the trendline + margin, remove the data point
        counter = len(current_data)
        new_data_objects = []

        for data in current_data:
            if data["fwhm"] < (krr.predict([[data["peak_pos"]]]) + self.fwhm_margin):
                new_data_objects.append(data)
                counter -= 1

        print(f"fit_fwhm_trendline() removed {counter} data points")
        
        return new_data_objects + baseline_data



    def peak_selection(self,
                       data_objects: list,
                       )-> list:
        
        """ Select data points based on the peak shape """

        counter = len(data_objects) - 1

        # loop through the data objects and fit the PL spectra
        new_data_objects = []
        for data in data_objects:
            if "spectrum" not in data:
                error = 0
            else:
                error = self.fit_PL_spectra(data["spectrum"])

            if error < self.peak_error_threshold:
                new_data_objects.append(data)
                counter -= 1

        print(f"peak_selection() removed {counter} data points")

        return new_data_objects



    def remove_outliers(self,
                        data_objects: list,
                        )-> list:
        """
            Remove data points that are outliers
            -> loo and calculate correlation between input and target 
            -> remove data points where the correlation improves significantly
        """

        counter = 0

        #TODO

        print(f"remove_outliers() removed {counter} data points")

        return data_objects


    def set_Cs_Pb_limit(self,
                        data_objects: list,
                        )-> list:
        """
            Set a limit for the Cs/Pb ratio TODO
        """
        pass


    def hard_FWHM_constraints(self, data_objects: list):
        """
            Apply hard constraints on the FWHM values
        """
        
        constraints = {"2": 13,
                       "3": 16,
                       "4": 25,
                       "5": 25,
                       "6": 25,
                       "7": 25,
                       "8": 24,
                       "9": 19}

        if self.mode == "EV":
            constraints = {key: self.nm_to_meV(value) for key, value in constraints.items()}


        # init return object
        new_data_objects = []

        for data in data_objects:

            ml = self.characterize_NPL(data["peak_pos"])

            if ml is not None:
                if data["fwhm"] <= constraints[str(ml)]:
                    new_data_objects.append(data)

        return new_data_objects






#### --------------------------- TARGET ADJ ---------------------------- ####

    def add_residual_targets(self, data_objects: list):
        """
            Convert the target values to residual values
        """

        # read the data
        targets  = np.array([data["y"] for data in data_objects])
        peak_pos = np.array([[data["peak_pos"]] for data in data_objects])
        

        # train a trendline model (it is critical that the trendline is not overfitting the data!)
        gp = GaussianProcess(training_data=peak_pos, targets=targets, kernel_type="RBF", model_type="GPRegression")
        gp.train()
        #gp.map_GP()


        # get the residuals
        residual_targets = [target - gp.predict(peak)[0][0] for target, peak in zip(targets, peak_pos)]

        # adjust the data objects
        for data, residual_target in zip(data_objects, residual_targets):
            print(f"residual_target: {residual_target}")
            data["y_res"] = residual_target
        
        return data_objects


    def add_residual_targets_avg(self, data_objects: list):

        """
            Convert the target values to residual values
            based on the average target for each antisolvent molecule
        """
        
        # write the average target values for each NPL ML number to a dictionary
        avg_targets  = {}
        for object in data_objects:
            peak_pos = object["peak_pos"]
            ml       = self.characterize_NPL(peak_pos)
            object["ml"] = ml

            if ml is not None:
                if ml in avg_targets:
                    avg_targets[ml].append(object["y"])
                else:
                    avg_targets[ml] = [object["y"]]

        for key, value in avg_targets.items():
            avg_targets[key] = np.mean(value)


        data_objects = [object for object in data_objects if object["ml"] is not None]

        # plot the average targets
        # plt.scatter([object["ml"] for object in data_objects], [object["y"] for object in data_objects], c="blue")
        # plt.scatter(avg_targets.keys(), avg_targets.values(), c="red", marker="x", s=100)
        # plt.show()
        
        

        # loop through the data objects and calculate the residual targets
        for object in data_objects:
            peak_pos = object["peak_pos"]
            ml = self.characterize_NPL(peak_pos)
            object["y_res"] = (object["y"] - avg_targets[ml])



        # avg. residual target
        #self.plot_avg_residuals(data_objects)
        
        return data_objects    


#### ---------------------------- HELPERS ------------------------------ ####
    

    def sort_by_target(self, data_objects: list)-> list:
        
        """ Sort data objects by target values """
        return sorted(data_objects, key = lambda x: x["y"])



    def nm_to_meV(self, wavelength) -> float:
        """
            Convert nm to meV
        """
        return 1239840 / wavelength



    def eV_to_nm(self, eV) -> float:
        """
            Convert eV to nm
        """
        return 1239.84 / eV



    def fit_PL_spectra(self, spectrum: tuple)-> float:

        """
            Fit the PL spectra with a Sigmoid*Voigt function
            --> return the avg. error of the fit
        """
        
        # fit the Voigt Sigmoid function
        #BOUNDS_SIG_VOIGT=  ([0, self.nm_to_meV(530), 0, 0, -0.5, 20], [1500, self.nm_to_meV(420), 100, 100, 0, 200])
        BOUNDS_SIG_VOIGT= ([0, 400, 1, 1, 0, 0], [100, 550, 30, 30, 10, 50])

        wavelength, norm_intensity = spectrum
        popt, pcov = curve_fit(self.Voigt_Sigmoid, wavelength, norm_intensity, bounds=BOUNDS_SIG_VOIGT, maxfev=100000)

        # calculate the error
        error = np.sum((self.Voigt_Sigmoid(wavelength, *popt) - norm_intensity) ** 2)/ len(wavelength)

        # visualize the fit
        # plt.plot(wavelength, norm_intensity)
        # plt.plot(wavelength, self.Voigt_Sigmoid(wavelength, *popt))
        # plt.show()

        return error


    def Voigt_Sigmoid(self, x, A, x_0, sigma, gamma, w, delta, ):
        """
            Product of a Voigt profile and a sigmoid function
            - > Voigt profile: convolution of a Gaussian and a Lorentzian
            - A: amplitude
            - x_0: peak position
            - w: sigmoid steepness
            - delta: sigmoid shift
        """

        return  A *  voigt_profile(x-x_0, sigma, gamma) *  ((1 / (1 + np.exp(-w*(x - x_0 - delta)))))


    def characterize_NPL(self, peak_pos)-> int:

        """
            Characterize the NPLs by their peak position
        """

        ml_dictionary = {"1": (402, 407),
                         "2": (430, 437),
                         "3": (457, 466),
                         "4": (472, 481),
                         "5": (484, 489),
                         "6": (491, 497),
                         "7": (498, 504),
                         "8": (505, 509),
                         "9": (510, 525),}
        

        if self.mode == "EV":
            peak_pos = self.eV_to_nm(peak_pos)
        
        for key, value in ml_dictionary.items():
            if value[0] <= peak_pos <= value[1]:
                return int(key)
        
        return None


### ---------------------------- VISUALIZATION ---------------------------- ####

    def plot_avg_residuals(self, data_objects: list):

        """
            Plot the average residuals for each molecule
        """

        molecules = list(set([object["molecule_name"] for object in data_objects]))
        avg_matrix = [[] for _ in range(len(molecules))]
        for object in data_objects:
            molecule = object["molecule_name"]
            avg_matrix[molecules.index(molecule)].append(object["y_res"])
        
        avg_residuals = [np.mean(residuals) for residuals in avg_matrix]
        standard_dev  = [np.std(residuals) for residuals in avg_matrix]

        # plot histogram
        plt.bar(molecules, avg_residuals, yerr=standard_dev)
        plt.show()


        