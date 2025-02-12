""" 
    Project:     synthesizer
    File:        Synthesizer.py
    Description: Defines the Synthesizer class for the optimization of NPL synthesis 
                 parameters using Gaussian Processes and Bayesian Optimization
    Author:      << github.com/leoluber >> 
    License:     MIT
"""




import numpy as np
import time
from datetime import date
import sys
import os
from typing import Literal
from GPyOpt.methods import BayesianOptimization


# custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from package.src.Datastructure import Datastructure
from package.src.GaussianProcess import GaussianProcess
from package.plotting.Plotter import Plotter






class Synthesizer:



    """ Optimizer Class for perovskite NPL synthesis parameters

    Optimization Module for Synthesis Parameter Recommendations for Perovskite 
    NPL synthesis; Samples synthesis parameters for a given NPL type while 
    optimizing the target values specified in the objectife function.

    
    MODULES
    -------
    - models:           Gaussian Process Regression (GPy)
    - optimization:     Bayesian Optimization (GPyOpt)

    
    ARGS
    ----
    - molecule (str):         molecule for the synthesis (e.g. "Methanol")
    - iterations (int):       max. number of optimization iterations
    - peak (int):             target peak position
    - ion (str):              NPL type (default is "CsPbI3")
    - obj (list):             properties included in the objective function
                              ("peak_pos", "plqy", "fwhm", "poly",)
    - c_Pb_fixed (float):     fixed Pb concentration (default is None)
    - V_As_fixed (float):     fixed As volume (default is None)
    - V_Cs_fixed (float):     fixed Cs volume (default is None)
    - Cs_Pb_opt (bool):       Cs/Pb ratio optimization (default is False)
    - Cs_As_opt (bool):       Cs/As ratio optimization (default is False)
    - c_Pb_max (float):       maximum Pb concentration (default is None)

    
    USAGE
    -----
    >>> synthesizer      = Synthesizer(...)
    >>> opt_x, opt_delta = synthesizer.optimize_NPL()
    >>> results          = synthesizer.results
    >>> synthesizer.print_results(results["results_string"], results["input_PLQY"], ... )

    """





    def __init__(
                 self,
                 molecule : str,
                 data_path : str,
                 spectral_path : str,
                 iterations : int,
                 peak : int,
                 ion : Literal["CsPbI3", "CsPbBr3"] = "CsPbBr3",
                 obj : list =       ["peak_pos", "fwhm",],
                 weights : list =   {"peak_pos": 10, 
                                     "poly": 10,
                                     "plqy": 5,
                                     "fwhm": 1,},
                 c_Pb_fixed =       None,
                 V_Cs_fixed =       None,
                 Cs_Pb_opt =        False,
                 Cs_As_opt =        False,
                 c_Pb_max =         None,
                 c_Cs_fixed =       None,
                 V_As_max =         5000,
                 add_baseline =     True,
                ):
        

        # set the parameters
        self.obj = obj
        self.weights = weights
        self.molecule =         molecule
        self.ion =              ion
        self.Cs_Pb_opt =        Cs_Pb_opt
        self.Cs_As_opt =        Cs_As_opt
        self.add_baseline =     add_baseline
        self.data_path =        data_path
        self.spectral_path =    spectral_path
        self.iterations =       iterations


        # initialize datastructure and encoding for current molecule
        self.datastructure =    self.get_datastructure()
        self.molecule_names =   self.datastructure.molecule_names
        self.encoding =         self.datastructure.encode(self.molecule)
        

        # results dictionary
        self.results = None


        # target peak position
        self.peak = peak


        # specify the parameters used for the optimization
        # TODO: refactor this to a more general approach
        if self.ion == "CsPbBr3":
            self.ratios =             ["AS_Pb_ratio", "Cs_Pb_ratio",]
            self.parameters_PEAK =    self.ratios 
            self.parameters =         self.parameters_PEAK + ["V (antisolvent)", "c (PbBr2)",  "c (Cs-OA)", "V (PbBr2 prec.)","V (Cs-OA)",]

            self.parameters_opt = [parameter for parameter in self.parameters if parameter not in self.ratios]
            self.parameters_opt_PEAK = [parameter for parameter in self.parameters_PEAK if parameter not in self.ratios]
            self.total_parameters = self.parameters_opt 
    
        if self.ion == "CsPbI3":
            self.ratios =             ["Cs_Pb_ratio",]
            self.parameters_PEAK =    self.ratios # + ["Pb/I" , "V (Cs-OA)", "t_Rkt",]
            self.parameters =         self.parameters_PEAK # + ["Centrifugation time [min]", "Centrifugation speed [rpm]",]

            self.parameters_opt =         [parameter for parameter in self.parameters if parameter not in self.ratios]
            self.parameters_opt_PEAK =    [parameter for parameter in self.parameters_PEAK if parameter not in self.ratios] 
            self.total_parameters =       self.parameters_opt + ["V (antisolvent)", "c (PbBr2)",  "c (Cs-OA)", "V (PbBr2 prec.)","V (Cs-OA)",]


        # extra constraints
        self.c_Pb_max = c_Pb_max
        self.V_As_max = V_As_max
        self.c_Pb_fixed = c_Pb_fixed
        self.c_Cs_fixed = c_Cs_fixed
        self.V_Cs_fixed = V_Cs_fixed

        
        # generate the limits for the optimization
        self.limits = self.get_limits()


        # train the models for the given objectives
        self.NPL_model  =           self.train_GP(self.parameters_PEAK, target = "peak_pos")
        
        if "plqy" in self.obj:
            self.PLQY_model =       self.train_GP(self.parameters, target = "plqy")

        if "fwhm" in self.obj:
            self.FWHM_model =       self.train_GP(self.parameters, target = "fwhm")

        if "poly" in self.obj:
            self.POLY_model =       self.train_GP(self.parameters, target = "polydispersity")


        # display the trained models to check for overfitting etc. 
        # (if the number of dimensions is sufficiently small)
        self.plotter = Plotter(self.datastructure.processed_file_path)
        self.plotter.plot_data( "AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", 
                              kernel= self.NPL_model, molecule= self.molecule,)
        



### ------------------------- OPTIMIZATION ------------------------- ###


    def optimize_NPL(self) -> tuple:


        """ Optimize for the selected NPL type using the GPyOpt library

        Minimizes the distance to the perfect peak position as well as
        the specified objectives (e.g. FWHM, PLQY, ...) using Bayesian 
        Optimization from the GPyOpt library.

        RETURNS
        -------
        tuple:  (optimal parameters, optimal loss)

        """

        # update
        print(f"\noptimizing NPL ...\n")


        # define the optimization domain from the limits dictionary
        bounds = [{'name': parameter, 
                   'type': 'continuous', 
                   'domain': (self.limits[parameter][0], 
                              self.limits[parameter][1])} 
                              for parameter in self.total_parameters]

        # optimize
        optimizer = BayesianOptimization(f = self.objective_function, domain = bounds)
        optimizer.run_optimization(max_iter = self.iterations)


        self.results = self.return_results(optimizer.x_opt)

        return optimizer.x_opt, optimizer.fx_opt



    def objective_function(self, x) -> float:

        """ Objective function for the optimization

        The objective function for the optimization problem, which is
        minimized by the Bayesian Optimization algorithm. The function
        calculates the loss from the current input parameters and the
        predictions from the Gaussian Process models.
        (note: the ratios are not part of the input parameters, but
        are calculated directly from the input parameters)

        ARGS
        ----
        x (np.array):  input parameters

        RETURNS
        -------
        float:  loss value

        """

        # extract the parameters from the input array
        c_Pb_norm, c_Cs_norm, V_Pb_norm, V_Cs_norm =  x[0][-4], x[0][-3], x[0][-2], x[0][-1]
        V_AS_norm = x[0][-5]

        # denormalize the parameters to calculate the ratios
        V_As = self.datastructure.denormalize(V_AS_norm, "V (antisolvent)")
        V_Pb = self.datastructure.denormalize(V_Pb_norm, "V (PbBr2 prec.)")
        V_Cs = self.datastructure.denormalize(V_Cs_norm, "V (Cs-OA)")
        c_Pb = self.datastructure.denormalize(c_Pb_norm, "c (PbBr2)") 
        c_Cs_norm = self.datastructure.denormalize(c_Cs_norm, "c (Cs-OA)")

        As_Pb_ratio = V_As *self.datastructure.densities[self.molecule] / (c_Pb * V_Pb * 10000) 
        Cs_Pb_ratio = (c_Cs_norm * V_Cs) / (c_Pb * V_Pb)


        # hard constraints 
        # TODO: refactor this to a more general approach
        if Cs_Pb_ratio < 0.2: return 1000
        if Cs_Pb_ratio > 0.4: return 1000
        #if Cs_Pb_ratio < 0.15: return 1000
        #if As_Pb_ratio > 0.8: return 1000


        # get the molecule encoding
        input = np.array(self.encoding)

        # construct the base input from ratios
        if self.ion == "CsPbBr3":
            input  = np.append(input, As_Pb_ratio)
        base_input = np.append(input, Cs_Pb_ratio)    
        
        # construct the input for the NPL model
        NPL_input = np.append(base_input, x[0][:len(self.parameters_opt_PEAK)])

        # construct the input for all other models
        total_input = np.append(base_input, x[0])
        print(f"total_input: {total_input}")

        # predictions
        NPL =   self.NPL_model.predict(NPL_input)[0][0]

        if "fwhm" in self.obj:
            FWHM =  self.FWHM_model.predict(total_input)[0][0]

        if "plqy" in self.obj:
            PLQY =  self.PLQY_model.predict(total_input)[0][0]

        if "poly" in self.obj:
            POLY =  self.POLY_model.predict(total_input)[0][0]
        


        # construct the objective function
        # weights are fixed; note that the PLQY is subtracted!
        output = abs(NPL[0] - self.peak) * self.weights["peak_pos"]

        if "fwhm" in self.obj:
            output += FWHM[0] * self.weights["fwhm"]
            print(f"FWHM: {FWHM * self.weights["fwhm"]}")

        if "plqy" in self.obj:
            output -= PLQY[0] * self.weights["plqy"]
            print(f"PLQY: {PLQY[0] * self.weights["plqy"]}")

        if "poly" in self.obj:
            output += POLY[0] * self.weights["poly"]
            print(f"POLY: {POLY[0] * self.weights["poly"]}")
                

        if self.Cs_Pb_opt:
            output += Cs_Pb_ratio * 20
        
        # print the current NPL value and ratios (for real-time feedback)
        print(f"NPL: {NPL}, Cs_Pb: {Cs_Pb_ratio}, As_Pb: {As_Pb_ratio}, loss:{output}, FWHM: {FWHM}")

        return output




### ----------------------------- INIT ------------------------------ ###

    def train_GP(self, parameter_selection, target) -> GaussianProcess:

        """ Train a Gaussian Process model on the given data 
        
        ARGS
        ----
        - parameter_selection (list):  list of parameters for the model
        - target (str):                target for the model

        RETURNS
        -------
        GaussianProcess:  trained GP model

        """

        inputs, targets, df = self.datastructure. \
                            get_training_data(training_selection = parameter_selection, 
                                              target = target, encoding = True, 
                                              remove_baseline = (target != "peak_pos"))
    
        
        gp = GaussianProcess(training_data = inputs,
                            targets = targets, 
                            kernel_type = "EXP",
                            )
        gp.train()

        # LOO cross validation to check model performance
        #gp.leave_one_out_cross_validation(inputs, targets,)
        #gp.regression_plot()

        return gp



    def get_limits(self) -> dict:

        """ Defines the limits for the NPL synthesis parameters

        RETURNS
        -------
        dict:  dictionary with the parameter limits

        """

        # standard limits are [0, 1] due to the normalization
        limits = {}
        for parameter in self.total_parameters:
            limits[parameter] = [0, 1]
        

        # local normalization function
        Norm = lambda value, parameter_string: ((value - self.datastructure.max_min[parameter_string][1]) 
                                                / (self.datastructure.max_min[parameter_string][0] 
                                                   - self.datastructure.max_min[parameter_string][1]))
        

        # if a custom limit is set, we adjust the limits accordingly
        if self.V_As_max is not None:
            max_V_As = self.V_As_max
            norm_max_V_As   = Norm(max_V_As, "V (antisolvent)")
            limits["V (antisolvent)"] = [0, norm_max_V_As]

        if self.c_Pb_fixed is not None:
            norm_c_Pb_fixed = Norm(self.c_Pb_fixed, "c (PbBr2)")
            limits["c (PbBr2)"] = [norm_c_Pb_fixed, norm_c_Pb_fixed]

        if self.c_Cs_fixed is not None:
            norm_c_Cs_fixed = Norm(self.c_Cs_fixed, "c (Cs-OA)")
            limits["c (Cs-OA)"] = [norm_c_Cs_fixed, norm_c_Cs_fixed]

        elif self.c_Pb_max is not None:
            max_c_Pb = self.c_Pb_max
            norm_max_c_Pb   = Norm(max_c_Pb, "c (PbBr2)")
            limits["c (PbBr2)"] = [0, norm_max_c_Pb]

        if self.V_Cs_fixed is not None:
            norm_V_Cs_fixed = Norm(self.V_Cs_fixed, "V (Cs-OA)")
            limits["V (Cs-OA)"] = [norm_V_Cs_fixed, norm_V_Cs_fixed]


        return limits
    


    def get_datastructure(self) -> Datastructure:

        """ Initialize the datastructure for the current molecule

        RETURNS
        -------
        Datastructure:  datastructure object

        """
        
        datastructure = Datastructure(synthesis_file_path = self.data_path,
                                        spectral_file_path  = self.spectral_path,
                                        monodispersity_only = True,
                                        molecule = "all",
                                        P_only = True,
                                        add_baseline = self.add_baseline,
                                        )
        
        datastructure.read_synthesis_data()

        return datastructure
            



### ------------------------ PRINTING AND TESTING ------------------------ ###


    def return_results(self, x) -> dict:

        """ Generates and returns the results from the optimization

        ARGS
        ----
        x (np.array):  optimal parameters

        RETURNS
        -------
        dict:  results dictionary

        """

        results = {}


        # get ratios (denormalize the parameters first)
        V_As = self.datastructure.denormalize(x[-5], "V (antisolvent)")
        c_Pb = self.datastructure.denormalize(x[-4], "c (PbBr2)")
        c_Cs = self.datastructure.denormalize(x[-3], "c (Cs-OA)")
        V_Pb = self.datastructure.denormalize(x[-2], "V (PbBr2 prec.)")
        V_Cs = self.datastructure.denormalize(x[-1], "V (Cs-OA)")

        results["As_Pb_ratio"] =  (V_As  * self.datastructure.densities[self.molecule] 
                                   / (c_Pb * V_Pb * 10000))
        results["Cs_Pb_ratio"] =  (c_Cs * V_Cs) / (c_Pb * V_Pb)


        # input
        input = np.array(self.encoding)
        if self.ion == "CsPbBr3":
            input  = np.append(input, results["As_Pb_ratio"])
        input = np.append(input, results["Cs_Pb_ratio"])  

        # predictions
        results["pred_peak"] = self.NPL_model.predict(input)[0][0]  


        # denormalize the parameters
        results["results_string"] = []
        denorm = {}

        for i, parameter in enumerate(self.total_parameters):
            denorm[parameter] = self.datastructure.denormalize(x[i], parameter)
            results["results_string"].append(f"{parameter} :  {denorm[parameter]}")

        return results



    def print_results(self, results_string, peak_pos,
                      As_Pb_ratio, Cs_Pb_ratio):
        
        """ Print the resulting suggestion to a file 

        Asks the user if the results should be printed to a file and
        then writes the results to a file in the output directory.
        
        ARGS
        ----
        results_string (str):  string with the optimized parameters
        input_NPL (list):      input parameters for the NPL model
        As_Pb_ratio (float):   As/Pb ratio
        Cs_Pb_ratio (float):   Cs/Pb ratio

        """



        print("----------------------------------------------------")
        print(f"RESULTS: \n")
        print(f"{results_string}\n")
        print(f"Peak position: {peak_pos}")
        print(f"As_Pb_ratio (old): {As_Pb_ratio}")
        print(f"Cs_Pb_ratio (old): {Cs_Pb_ratio}")
        print("----------------------------------------------------")

        print("Do you want to print the results? (y/n)")
        if input() == "y":

            print("writing to file ...")

            # write to file
            with open("output/suggestions.txt", "a") as file:

                file.write(f"\n")
                file.write(f"------------------------------------   {str(date.today())}   ---------------------------------\n")
                file.write(f"L-SYNTH 2.0-{np.random.randint(10000)}\n")
                file.write(f"THE SYNTHESIZER RECOMMENDS: \n")
                file.write(f"\n")
                file.write(f"When using -{self.molecule}- as an antisolvent \n")
                file.write(f"to make {self.ion} NPLs around {self.peak}nm:\n")
                file.write(f"Choose the following synthesis parameters:\n")
                file.write(f"{results_string}\n")

                file.write(f"with the following As/Pb ratio: {As_Pb_ratio}\n")
                file.write(f" and the following Cs/Pb ratio: {Cs_Pb_ratio}\n")

                file.write(f"\n")
                file.write(f"Predicted peak position: {peak_pos}\n")

                file.write(f"\n")
                file.write(f"DEV:\n")

                # write settings
                file.write(f"SETTINGS:  Obj-> {self.obj}, iterations-> {self.iterations},  peak-> {self.peak}\n")
                file.write(f"molecule-> {self.molecule},  add_baseline-> {self.add_baseline}\n")

                file.write(f"\n")
                file.write(f"THANK YOU FOR USING THE SYNTHESIZER MODULE!\n")
            
            print("done writing to file ...")


        return 0

    


    def print_logo(self):

        """ Print the logo to the console
        -> ASCII image from ascii-art-generator.org

        """

        # read in from file (ascii-art.txt)
        with open("data/ascii-art.txt", "r") as file:
            for line in file:
                print(line, end="")
                time.sleep(0.02)	

        print("\n")
    
