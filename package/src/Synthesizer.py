""" 
    File:        Synthesizer.py
    Project:     Synthesizer: Chemistry-Aware Machine Learning for 
                 Precision Control of Nanocrystal Growth 
                 (Henke et al., Advanced Materials 2025)
    Description: Defines the Synthesizer class for the optimization of NC synthesis 
                 parameters using Gaussian Processes and Bayesian Optimization
    Author:      << github.com/leoluber >> 
    License:     MIT
"""



# ------------------------
import numpy as np
from datetime import date
import sys
import os
from GPyOpt.methods import BayesianOptimization
# ------------------------



# custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from package.src.Datastructure import Datastructure
from package.src.GaussianProcess import GaussianProcess
from package.plotting.Plotter import Plotter



class Synthesizer:


    """ Optimizer Class for perovskite NC synthesis parameters

    Optimization Module for Synthesis Parameter Recommendations for Perovskite 
    NC synthesis; Samples synthesis parameters for a given NC PL peak pos. while 
    also optimizing the target values specified in the objective function.

    MODULES
    -------
    - models:           Gaussian Process Regression (GPy)
    - optimization:     Bayesian Optimization (GPyOpt)

    ARGS
    ----
    - molecule (str):         molecule for the synthesis (e.g. "Methanol")
    - iterations (int):       max. number of optimization iterations
    - peak (int):             target peak position
    - obj (list):             properties included in the objective function
                              ("peak_pos", "plqy", "fwhm",)
    - V_As_fixed (float):     fixed As volume (default is None)
    - V_Cs_fixed (float):     fixed Cs volume (default is None)
    - c_Pb_max (float):       maximum Pb concentration (default is None)

    
    USAGE
    -----
    >>> synthesizer      = Synthesizer(...)
    >>> opt_x, opt_delta = synthesizer.optimize_NC()
    >>> results          = synthesizer.results
    >>> synthesizer.print_results(results["results_string"], results["input_PLQY"], ... )

    """




    def __init__(
                 self,
                 molecule : str,
                 data_path : str,
                 iterations : int,
                 peak : int,

                 # objectives to optimize
                 obj : list =       ["peak_pos", "fwhm",], #plqy

                 # here the weights for the different objectives can be adjusted
                 weights : list =   {"peak_pos": 10, 
                                     "plqy": 5,
                                     "fwhm": 10,},

                 # in some cases, it might be useful to fix certain parameters
                 # (add more if needed; see get_limits() function)
                 V_Cs_fixed =       None,
                 c_Cs_fixed =       None,

                 # some experimental constraints (e.g. max. volumes) can be added here
                 V_As_max =         5000,
                 V_Pb_max =         None,
                 add_baseline =     True,

                ):
        

        # set the parameters
        self.obj =              obj
        self.weights =          weights
        self.molecule =         molecule
        self.add_baseline =     add_baseline
        self.data_path =        data_path
        self.iterations =       iterations

        # initialize datastructure and encoding for current molecule
        self.datastructure =    self.get_datastructure()
        self.encoding =         self.datastructure.encode(self.molecule)
        self.encoding_dim =     len(self.encoding)

        # initialize for later use
        self.selection_dataframe = None
        self.results = None

        # target peak position
        self.peak = peak

        ### ------------ TODO: ADJUST FOR OTHER APPLICATIONS ------------ ###
        # specify the parameters used for the optimization (the order matters!)

        # NOTE: as the ratio space fully defines the peak position, we only
        #       need to include those in the PEAK model; for FWHM and PLQY
        #       we include the full parameter space (see publication for details)

        self.parameters_PEAK =      ["AS_Pb_ratio", "Cs_Pb_ratio"]
        self.synthesis_parameters = ["V (antisolvent)", "c (PbBr2)",  "c (Cs-OA)", "V (PbBr2 prec.)","V (Cs-OA)",]
        self.total_parameters =     self.parameters_PEAK + self.synthesis_parameters
        ### ------------------------------------------------------------- ###

        # constraints
        self.V_As_max = V_As_max
        self.V_Pb_max = V_Pb_max
        self.c_Cs_fixed = c_Cs_fixed
        self.V_Cs_fixed = V_Cs_fixed
        
        # generate the limits for the optimization (depending on the constraints)
        self.limits = self.get_limits()

        # train the models for the given objectives (peak position, FWHM, PLQY, ...)
        self.PEAK_model  =           self.train_GP(self.parameters_PEAK, target = "peak_pos")

        if "plqy" in self.obj:
            self.PLQY_model =       self.train_GP(self.total_parameters, target = "plqy")
        if "fwhm" in self.obj:
            self.FWHM_model =       self.train_GP(self.total_parameters, target = "fwhm")


        # display the trained model to check for overfitting etc. 
        self.plotter = Plotter(self.datastructure.processed_file_path, selection_dataframe= self.selection_dataframe)
        self.plotter.plot_data( "AS_Pb_ratio", "Cs_Pb_ratio", "peak_pos", 
                              kernel= self.PEAK_model, molecule= self.molecule,)
        



### ------------------------- OPTIMIZATION ------------------------- ###


    def optimize_NC(self) -> tuple:


        """ Optimize for the selected NC type using the GPyOpt library

        Minimizes the distance to the perfect peak position as well as
        the specified objectives (e.g. FWHM, PLQY, ...) using Bayesian 
        Optimization from the GPyOpt library.

        RETURNS
        -------
        tuple:  (optimal parameters, optimal loss)

        """

        # update
        print(f"\noptimizing NC ...\n")


        # define the optimization domain from the limits dictionary
        bounds = [{'name': parameter, 
                   'type': 'continuous', 
                   'domain': (self.limits[parameter][0], 
                              self.limits[parameter][1])} 
                              for parameter in self.synthesis_parameters]

        # optimize
        optimizer = BayesianOptimization(f = self.objective_function, domain = bounds)
        optimizer.run_optimization(max_iter = self.iterations)


        self.results = self.return_results(optimizer.x_opt)

        return optimizer.x_opt, optimizer.fx_opt


    def objective_function(self, x) -> float:

        """ Objective function for the optimization

        The objective function for the optimization problem, which is
        minimized by the Bayesian Optimization algorithm. 

        The input "x" represents a point sampled from parameter space represented by
        volumes and concentrations of the synthesis components. Together with the
        ratios (As/Pb and Cs/Pb; calculated from the sampled parameters)
        they are used as input for the trained Gaussian Process models.
        The function calculates the loss from the current input parameters and the
        predictions from the Gaussian Process models.

        NOTE: denormalization of the parameters is done here, as the ratios
              need to be calculated from the actual values; we avoid sampling in ratio
              space directly as this would lead to a non-uniform sampling of the
              parameter space.
        """

        ### --- CALCULATE RATIOS --- ###

        """ NOTE: the ratios are not part of the input parameters, but
                  are calculated from the input parameters here
                  --> this whole section needs to be adjusted for each separate application
                  --> the indices used below are specific to the order of the parameters
                      defined in self.total_parameters and self.parameters_PEAK
                  """
        
        # denormalize the parameters to calculate the ratios
        V_As = self.datastructure.denormalize(x[0][-5], "V (antisolvent)")
        V_Pb = self.datastructure.denormalize(x[0][-2], "V (PbBr2 prec.)")
        V_Cs = self.datastructure.denormalize(x[0][-1], "V (Cs-OA)")
        c_Pb = self.datastructure.denormalize(x[0][-4], "c (PbBr2)") 
        c_Cs_norm = self.datastructure.denormalize(x[0][-3], "c (Cs-OA)")

        # calculate the ratios (note the rescaling of the As/Pb ratio by 10000; see publication)
        As_Pb_ratio = V_As *self.datastructure.concentrations[self.molecule] / (c_Pb * V_Pb * 10000) 
        Cs_Pb_ratio = (c_Cs_norm * V_Cs) / (c_Pb * V_Pb)
        ### --------------------------- ###


        # hard constraints can be added here (e.g. max. ratios), though this is not recommended
        #if Cs_Pb_ratio > 1:  return 1000


        ### --- Constructing the Input --- ###
        # start with the molecule encoding
        input = np.array(self.encoding)

        # construct the PEAK input by adding the ratios
        base_input  = np.append(input, As_Pb_ratio)
        PEAK_input = np.append(base_input, Cs_Pb_ratio)    
        print(f"PEAK input: {PEAK_input}")

        # construct the input for all other models
        total_input = np.append(PEAK_input, x[0])
        print(f"total input: {total_input}")

        ### --- PREDICTIONS FOR CURRENT INPUT --- ###
        PEAK =   self.PEAK_model.predict(PEAK_input)[0][0]

        if "fwhm" in self.obj:
            FWHM =  self.FWHM_model.predict(total_input)[0][0]
        if "plqy" in self.obj:
            PLQY =  self.PLQY_model.predict(total_input)[0][0]


        ### --- CALCULATE THE LOSS --- ###
        # NOTE: PLQY is subtracted here as we want to maximize it
        #       while the other objectives are minimized
        #       the weights for the different objectives can be adjusted in self.weights

        output = abs(PEAK[0] - self.peak) * self.weights["peak_pos"]

        if "fwhm" in self.obj:
            output += FWHM[0] * self.weights["fwhm"]
            #print(f"FWHM: {FWHM * self.weights["fwhm"]}")

        if "plqy" in self.obj:
            output -= PLQY[0] * self.weights["plqy"]
            #print(f"PLQY: {PLQY[0] * self.weights["plqy"]}")

        # print the current PEAK value and ratios (for real-time feedback)
        print(f"PEAK: {PEAK}, Cs_Pb: {Cs_Pb_ratio}, As_Pb: {As_Pb_ratio}, loss:{output}, FWHM: {FWHM}")

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
     
        if target == "peak_pos":
            self.selection_dataframe = df  # for plotting
        
        gp = GaussianProcess(training_data = inputs,
                            targets = targets, 
                            kernel_type = "EXP",
                            )
        gp.train()

        ### --- LOO cross validation to check model performance --- ###
        #gp.leave_one_out_cross_validation(inputs, targets,)
        #gp.regression_plot()

        return gp



    def get_limits(self) -> dict:

        """ Defines the limits for the NC synthesis parameters
            Gneral limits are [0, 1] due to the normalization
            however, custom limits can be set for certain parameters

        RETURNS
        -------
        dict:  dictionary with the parameter limits

        """

        # standard limits are [0, 1] due to the normalization
        limits = {}
        for parameter in self.synthesis_parameters:
            limits[parameter] = [0, 1]

        # local normalization function using the datastructure max/min values
        Norm = lambda value, parameter_string: ((value - self.datastructure.max_min[parameter_string][1]) 
                                                / (self.datastructure.max_min[parameter_string][0] 
                                                   - self.datastructure.max_min[parameter_string][1]))
        

        # if a custom limit is set, we adjust accordingly
        # NOTE: we need to normalize the custom limits as well
        if self.V_As_max is not None:
            max_V_As = self.V_As_max
            norm_max_V_As   = Norm(max_V_As, "V (antisolvent)")
            limits["V (antisolvent)"] = [0, norm_max_V_As]

        if self.V_Pb_max is not None:
            max_V_Pb = self.V_Pb_max
            norm_max_V_Pb   = Norm(max_V_Pb, "V (PbBr2 prec.)")
            limits["V (PbBr2 prec.)"] = [0, norm_max_V_Pb]

        if self.c_Cs_fixed is not None:
            norm_c_Cs_fixed = Norm(self.c_Cs_fixed, "c (Cs-OA)")
            limits["c (Cs-OA)"] = [norm_c_Cs_fixed, norm_c_Cs_fixed]

        if self.V_Cs_fixed is not None:
            norm_V_Cs_fixed = Norm(self.V_Cs_fixed, "V (Cs-OA)")
            limits["V (Cs-OA)"] = [norm_V_Cs_fixed, norm_V_Cs_fixed]

        return limits
    


    def get_datastructure(self) -> Datastructure:

        """ Initialize the datastructure for the current molecule
            and read in the synthesis data
        """
        
        datastructure = Datastructure(synthesis_file_path = self.data_path,
                                        monodispersity_only = True,
                                        molecule = "all",
                                        P_only = True,
                                        add_baseline = self.add_baseline,
                                        encoding = "geometry",
                                        )
        
        datastructure.read_synthesis_data()

        return datastructure
            



### ------------------------ PRINTING AND TESTING ------------------------ ###


    def return_results(self, x) -> dict:

        """ Generates and returns the results from the optimization

        NOTE: again, the calculations here are specific to the perovskite NC
              synthesis in Henke et al. and need to be adjusted for other applications
        """

        results = {}

        # get ratios (denormalize the parameters first; beware the indices!)
        V_As = self.datastructure.denormalize(x[-5], "V (antisolvent)")
        c_Pb = self.datastructure.denormalize(x[-4], "c (PbBr2)")
        c_Cs = self.datastructure.denormalize(x[-3], "c (Cs-OA)")
        V_Pb = self.datastructure.denormalize(x[-2], "V (PbBr2 prec.)")
        V_Cs = self.datastructure.denormalize(x[-1], "V (Cs-OA)")

        results["As_Pb_ratio"] =  (V_As  * self.datastructure.concentrations[self.molecule] 
                                   / (c_Pb * V_Pb * 10000))
        results["Cs_Pb_ratio"] =  (c_Cs * V_Cs) / (c_Pb * V_Pb)

        # input
        input = np.array(self.encoding)
        input  = np.append(input, results["As_Pb_ratio"])
        input = np.append(input,  results["Cs_Pb_ratio"])  

        # predict the peak position again to check the result
        results["pred_peak"] = self.PEAK_model.predict(input)[0][0]  

        # denormalize the parameters
        results["results_string"] = []
        denorm = {}

        for i, parameter in enumerate(self.synthesis_parameters):
            denorm[parameter] = self.datastructure.denormalize(x[i], parameter)
            results["results_string"].append(f"{parameter} :  {denorm[parameter]}")

        return results



    def print_results(self, results_string, peak_pos,
                      As_Pb_ratio, Cs_Pb_ratio):
        
        """ Print the resulting suggestion to a file 

        Asks the user if the results should be printed to a file and
        then writes the results to a file in the output directory.
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
                file.write(f"L-SYNTH -{np.random.randint(10000000)}\n")
                file.write(f"THE SYNTHESIZER RECOMMENDS: \n")
                file.write(f"\n")
                file.write(f"When using -{self.molecule}- as an antisolvent \n")
                file.write(f"to make CsPbBr3 NCs around {self.peak} nm:\n")
                file.write(f"Choose the following synthesis parameters:\n")
                file.write(f"{results_string}\n")
                file.write(f"with the following As/Pb ratio: {As_Pb_ratio}\n")
                file.write(f" and the following Cs/Pb ratio: {Cs_Pb_ratio}\n")
                file.write(f"\n")
                file.write(f"Predicted peak position: {peak_pos}\n")
                file.write(f"\n")
                file.write(f"DEV:\n")

                # write settings to file for future reference
                file.write(f"SETTINGS:  Obj-> {self.obj}, iterations-> {self.iterations},  peak-> {self.peak}\n")
                file.write(f"molecule-> {self.molecule},  add_baseline-> {self.add_baseline}\n")
                file.write(f"\n")
                file.write(f"THANK YOU FOR USING THE SYNTHESIZER MODULE!\n")
            
            print("done writing to file ...")

        return 0

