r""" Bayesian Optimization for Perovskite NPL synthesis parameters based on 
    Gaussian Process and Kernel Ridge Regression models"""
    # << github.com/leoluber >> 


import numpy as np
import time
from GPyOpt.methods import BayesianOptimization
from datetime import date
from typing import Literal

# custom modules
from Datastructure import *
from helpers import *
from KRR import Ridge
from GaussianProcess import GaussianProcess
from Preprocessor import Preprocessor



class Synthesizer:

    """ Optimizer Class

    Optimization Module with Synthesis Parameter Recommendations for Perovskite 
    NPL synthesis; Samples synthesis parameters for a given NPL type while 
    optimizing the target values (PLQY, FWHM, ...)

    BASICS
    ------
    - models:           Gaussian Process, Kernel Ridge Regression (GPy, scikit-learn)
    - datastructure:    Datastructure (custom)
    - optimization:     Bayesian Optimization (GPyOpt)

    PARAMETERS
    ----------
    - molecule:         molecule for the synthesis (e.g. "Methanol")
    - iterations:       max. number of optimization iterations
    - peak:             target peak position
    - obj:              properties included in the objective function
    - encoding_type:    one hot encoding or geometry encoding
    - model_type:       model type for the optimization (KRR, GP)
    - geometry:         geometry encoding of an UNKNOWN molecule; 
                        None if we predict a known molecule
    - c_Pb_fixed:       fixed Pb concentration (default is None)
    - V_As_fixed:       fixed As volume (default is None)
    - V_Cs_fixed:       fixed Cs volume (default is None)
    - Cs_Pb_opt:        Cs/Pb ratio optimization (default is False)
    - Cs_As_opt:        Cs/As ratio optimization (default is False)
    - c_Pb_max:         maximum Pb concentration (default is None)

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
                 iterations : int,
                 peak : int,
                 obj : str =    ["PEAK_POS", "PLQY", "FWHM"],
                 encoding_type: Literal["one_hot", "geometry"] = "one_hot",
                 model_type:    Literal["KRR", "GP"] = "GP",
                 geometry =     None,
                 c_Pb_fixed =   None,
                 V_As_fixed =   None,
                 V_Cs_fixed =   None,
                 Cs_Pb_opt =    False,
                 Cs_As_opt =    False,
                 c_Pb_max =     None,
                 V_As_max =     None,
                ):
        

        # set of parameters used in the objective function
        self.obj = obj
        self.molecule =         molecule
        self.Cs_Pb_opt =        Cs_Pb_opt
        self.Cs_As_opt =        Cs_As_opt


        # main parameters
        self.model_type =       model_type
        self.geometry =         geometry
        self.encoding_type =    encoding_type
        if self.encoding_type == "one_hot" and self.geometry is not None:
            raise ValueError("one hot encoding and geometry setting are not compatible")
        self.datastructure_NPL, self.datastructure_PLQY, self.datastructure_FWHM= self.get_datastructures()
        self.iterations =       iterations
        self.molecule_names =   self.datastructure_NPL.molecule_names
        self.encoding =         self.datastructure_NPL.encode(self.molecule, self.encoding_type)
        self.results = None


        # target peak position
        self.peak = peak


        # data
        self.inputs_NPL, self.peak_pos =   self.get_data(self.datastructure_NPL)
        self.inputs_PLQY, self.PLQY =  self.get_data(self.datastructure_PLQY)
        if "FWHM" in self.obj:
            self.inputs_FWHM, self.FWHM =  self.get_data(self.datastructure_FWHM)

        self.parameters_NPL  =      self.datastructure_NPL.synthesis_training_selection
        self.parameters_PLQY =  self.datastructure_PLQY.synthesis_training_selection
        if "FWHM" in self.obj:  
            self.parameters_FWHM =  self.datastructure_FWHM.synthesis_training_selection
        

        # extra constraints
        self.c_Pb_max = c_Pb_max
        self.V_As_max = V_As_max
        self.c_Pb_fixed = c_Pb_fixed
        self.V_As_fixed = V_As_fixed
        self.V_Cs_fixed = V_Cs_fixed
        self.limits = self.get_limits()


        # models
        if model_type == "KRR":
            self.NPL_model  =       self.train(self.inputs_NPL, self.peak_pos, 
                                               self.parameters_NPL)
        elif model_type == "GP":
            self.NPL_model  =       self.train_GP(self.inputs_NPL, self.peak_pos, 
                                                  self.parameters_NPL)
        else:
            raise ValueError("model type not recognized")

        if "PLQY" in self.obj:
            self.PLQY_model =       self.train(self.inputs_PLQY, self.PLQY, 
                                               self.parameters_PLQY)

        if "FWHM" in self.obj:
            self.FWHM_model =       self.train(self.inputs_FWHM, self.FWHM, 
                                               self.parameters_FWHM)


        # show the models for the user
        self.datastructure_NPL.plot_data(self.parameters_NPL[0], 
                                         self.parameters_NPL[1], 
                                         kernel = self.NPL_model,
                                         model = model_type ,  
                                         molecule = self.molecule)



### ------------------------- OPTIMIZATION ------------------------- ###


    def optimize_NPL(self) -> tuple:
        """
            Optimize for the selected NPL type using the GPyOpt library
            - minimizes the distance to the perfect peak position
            - returns the optimal parameters and the best peak position
        """

        print("optimizing NPL ...")
        print(f"\n")


        # define the optimization domain from the limits dictionary
        bounds = [{'name': parameter, 
                   'type': 'continuous', 
                   'domain': (self.limits[parameter][0], 
                              self.limits[parameter][1])} 
                              for parameter in self.parameters_PLQY[2:]]
        

        # defining the objective function
        def f(x):

            """
                The As_Pb ratio is calculated from the input parameters 
                since it is not included in the optimization itself
                - it is used here as an input for the NPL model (#MLs)
            """

            # calculate ratios from the current input parameters
            V_AS_norm, c_Pb_norm, V_Pb_norm, V_Cs_norm = x[0][0], x[0][1], x[0][2], x[0][3]
            V_As = self.datastructure_NPL.denormalize(V_AS_norm, "V (antisolvent)")
            V_Pb = self.datastructure_NPL.denormalize(V_Pb_norm, "V (PbBr2 prec.)")
            V_Cs = self.datastructure_NPL.denormalize(V_Cs_norm, "V (Cs-OA)")
            c_Pb = self.datastructure_NPL.denormalize(c_Pb_norm, "c (PbBr2)") 

            As_Pb_ratio = V_As *self.datastructure_NPL.densities[self.molecule] / (c_Pb * V_Pb * 10000) 
            Cs_Pb_ratio = (0.02 * V_Cs) / (c_Pb * V_Pb)
            Cs_As_ratio = (V_Cs) / (V_As)


            # hard constraints (not ideal, but simpler than adding them to the bounds)
            if Cs_Pb_ratio > 0.7: return 1000
            if As_Pb_ratio > 0.8: return 1000


            # add the encoded molecule to both inputs
            if self.geometry is None:  input = np.array(self.encoding)
            else:                      input = np.array(self.geometry)

            NPL_input =        np.append(np.append(input, As_Pb_ratio), Cs_Pb_ratio)    
            FWHM_PLQY_input =  np.append(NPL_input, x)

            if self.model_type == "KRR":
                NPL_input =  [value for value in NPL_input]
            elif self.model_type == "GP":	
                NPL_input =  NPL_input
            else:
                raise ValueError("model type not recognized")

            FWHM_PLQY_input = [value for value in FWHM_PLQY_input]



            # predictions
            if self.model_type == "KRR":
                NPL =   self.NPL_model.predict([NPL_input])

            elif self.model_type == "GP":
                NPL =   self.NPL_model.predict(NPL_input)[0][0]
                err =   self.NPL_model.predict(NPL_input)[1][0]
            else:
                raise ValueError("model type not recognized")
            
            # predictions for the other models
            if "FWHM" in self.obj:
                FWHM =  self.FWHM_model.predict([FWHM_PLQY_input])

            if "PLQY" in self.obj:
                PLQY =  self.PLQY_model.predict([FWHM_PLQY_input])
            

            # print the current NPL value and ratios (for real-time feedback)
            print(f"NPL: {NPL}, As_Pb: {As_Pb_ratio}, Cs_Pb: {Cs_Pb_ratio}")

            # plot the optimization process
            plt.scatter(NPL_input[-2], NPL_input[-1], c = NPL, vmin=450, vmax = 500)


            # construct the objective function
            output = abs(NPL - self.peak) * 10

            if "FWHM" in self.obj:
                output += FWHM[0]

            if "PLQY" in self.obj:
                output -= PLQY[0]*5
            
            if self.Cs_Pb_opt:
                output += Cs_Pb_ratio * 20

            if self.Cs_As_opt:
                output += Cs_As_ratio * 20

            if "UNCERTAINTY" in self.obj:
                output += (1000-err)/1000

            return output



        # optimize
        optimizer = BayesianOptimization(f = f, domain = bounds)
        optimizer.run_optimization(max_iter = self.iterations)

        # show the optimization process
        plt.colorbar() 
        plt.show()

        self.results = self.return_results(optimizer.x_opt)

        return optimizer.x_opt, optimizer.fx_opt




### -------------------------- INIT -------------------------- ###
    
    def train(self, inputs, targets, parameter_selection):
        """ Train a RKK model on the given data """

        rkk = Ridge(inputs, targets, 
                    parameter_selection, 
                    kernel_type= "laplacian", alpha=0.01, gamma=0.01)

        # optimize hyperparameters
        #print("finding hyperparameters ... ")

        #the RKK hyper_parameter optimization is defined in the KRR class
        #rkk.optimize_hyperparameters()

        rkk.fit()
        return rkk


    def train_GP(self, inputs, targets, parameter_selection):
        """ Train a Gaussian Process model on the given data """

        gp = GaussianProcess(training_data = np.array(inputs),
                            parameter_selection = parameter_selection,
                            targets = np.array(targets), 
                            kernel_type = "EXP",
                            model_type  = "GPRegression",
                            )
        gp.train()
        return gp


    def get_data(self, datastructure : Datastructure) -> tuple:
        """ Get data from the Datastructure object """

        data_objects = datastructure.get_data()

        # adding the residual targets to the data (-> see Datastructure.py)
        if datastructure.target in ["PLQY", "FWHM"]:
            ds = Preprocessor(mode = datastructure.wavelength_unit)
            data_objects = ds.add_residual_targets_avg(data_objects)


        inputs =  [data["encoding"] + data["total_parameters"] 
                   for data in data_objects]     # including one hot encoding
        

        # set the target values for the optimization
        if datastructure.target in ["PLQY", "FWHM"]:
            targets  = [data["y_res"] for data in data_objects]
        else:
            targets  = [data["y"] for data in data_objects]

        return inputs, targets



    def get_limits(self) -> dict:
        """
            Defines the limits for the NPL synthesis parameters from the peak range,
            returns them as a dictionary
        """

        limits = {}

        # exclude the one hot encoded molecule and ratios; +2 for the As/Pb and Cs/Pb ratios
        samples = [input[len(self.encoding)+2:] for input in self.inputs_PLQY]

        
        for i, parameter in enumerate(self.parameters_PLQY[2:]):
            limits[parameter] = [min([sample[i] for sample in samples]), 
                                 max([sample[i] for sample in samples])]
        
        
        # local normalization function
        Norm = lambda value, parameter_string: ((value - self.datastructure_NPL.max_min[parameter_string][1]) 
                                                / (self.datastructure_NPL.max_min[parameter_string][0] 
                                                   - self.datastructure_NPL.max_min[parameter_string][1]))
        
        if self.V_As_fixed is not None:
            norm_V_As_fixed = Norm(self.V_As_fixed, "V (antisolvent)")
            limits["V (antisolvent)"] = [norm_V_As_fixed, norm_V_As_fixed]
        elif self.V_As_max is not None:
            max_V_As = self.V_As_max
            norm_max_V_As   = Norm(max_V_As, "V (antisolvent)")
            limits["V (antisolvent)"] = [0, norm_max_V_As]

        if self.c_Pb_fixed is not None:
            norm_c_Pb_fixed = Norm(self.c_Pb_fixed, "c (PbBr2)")
            limits["c (PbBr2)"] = [norm_c_Pb_fixed, norm_c_Pb_fixed]
        elif self.c_Pb_max is not None:
            max_c_Pb = self.c_Pb_max
            norm_max_c_Pb   = Norm(max_c_Pb, "c (PbBr2)")
            limits["c (PbBr2)"] = [0, norm_max_c_Pb]

        if self.V_Cs_fixed is not None:
            norm_V_Cs_fixed = Norm(self.V_Cs_fixed, "V (Cs-OA)")
            limits["V (Cs-OA)"] = [norm_V_Cs_fixed, norm_V_Cs_fixed]


        return limits
    


    def get_datastructures(self) -> tuple:
        """
            Get the datastructures for the NPL, PLQY and FWHM models
            --> make adjustments here if necessary
        """

        ## - choosing the correct settings for the datastructures - ##
        # if we predict an unknown molecule, we need "molecule = "all"" to extrapolate to the 
        # unknown molecule using the geometric encoding TODO: change to fingerprint encoding

        if self.geometry is not None:
            pred_molecule = "all"
            encoding = "geometry"
            P = False
        else:
            pred_molecule = self.molecule
            encoding = self.encoding_type

        
        datastructure_NPL = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                                        target = "PEAK_POS",
                                        wavelength_unit= "NM",     
                                        wavelength_filter= [400, 550],
                                        encoding= encoding,
                                        monodispersity_only= True,
                                        molecule= pred_molecule,
                                        P_only= False,
                                        add_baseline= True,
                                        )
        
        datastructure_NPL.synthesis_training_selection = ["AS_Pb_ratio", "Cs_Pb_ratio"]	


        datastructure_PLQY = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv",
                                        target = "PLQY",
                                        wavelength_unit= "NM",         
                                        wavelength_filter= [400, 510],
                                        encoding= encoding,    
                                        monodispersity_only= True,  
                                        PLQY_criteria = True, 
                                        P_only= True,                      
                                        )
        datastructure_PLQY.synthesis_training_selection = ["AS_Pb_ratio", "Cs_Pb_ratio", "V (antisolvent)", 
                                                           "c (PbBr2)", "V (PbBr2 prec.)", "V (Cs-OA)"]


        datastructure_FWHM = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv",
                                        target = "FWHM",
                                        wavelength_unit= "NM",             
                                        wavelength_filter= [400, 550],
                                        monodispersity_only= True,
                                        encoding= encoding,
                                        PLQY_criteria = True,
                                        P_only= True,
                                        )
        datastructure_FWHM.synthesis_training_selection = ["AS_Pb_ratio", "Cs_Pb_ratio", "V (antisolvent)", 
                                                           "c (PbBr2)", "V (PbBr2 prec.)", "V (Cs-OA)"]
    

        return datastructure_NPL, datastructure_PLQY, datastructure_FWHM
            



### ------------------------ PRINTING AND TESTING ------------------------ ###

    def return_results(self, x) -> dict:
        """
            return the results from the optimization 
        """

        results = {}

        # get As/Pb ratio (denormalize the parameters first)
        V_As = self.datastructure_NPL.denormalize(x[0], "V (antisolvent)")
        c_Pb = self.datastructure_NPL.denormalize(x[1], "c (PbBr2)")
        V_Pb = self.datastructure_NPL.denormalize(x[2], "V (PbBr2 prec.)")
        V_Cs = self.datastructure_NPL.denormalize(x[3], "V (Cs-OA)")
        c_Cs = 0.02

        results["As_Pb_ratio"] =  (V_As  * self.datastructure_NPL.densities[self.molecule] 
                                   / (c_Pb * V_Pb * 10000))
        results["Cs_Pb_ratio"] =  (c_Cs * V_Cs) / (c_Pb * V_Pb)


        # get initial input that leads to the best plqy value
        encoding =        np.array(self.encoding)
        input_NPL =       np.append(encoding, results["As_Pb_ratio"])
        input_NPL =       np.append(input_NPL, results["Cs_Pb_ratio"])
        input_PLQY_FWHM = np.append(input_NPL, x)

        results["input_NPL"] =  [value for value in input_NPL]
        results["input_PLQY"] = [value for value in input_PLQY_FWHM]
        results["input_FWHM"] = [value for value in input_PLQY_FWHM]


        # denormalize the parameters
        results["results_string"] = []
        denorm = {}

        for i, parameter in enumerate(self.parameters_PLQY[2:]):
            denorm[parameter] = self.datastructure_PLQY.denormalize(x[i], parameter)
            results["results_string"].append(f"{parameter} :  {denorm[parameter]}")

        return results



    def print_results(self, results_string, input_PLQY, input_NPL, 
                      input_FWHM, As_Pb_ratio, Cs_Pb_ratio):
        """ Print the results to a file """

        # for GP
        if self.model_type == "GP":
            input_NPL = input_NPL
        else:
            input_NPL = [input_NPL]

        print("----------------------------------------------------")

        print(f"As_Pb_ratio (old): {As_Pb_ratio}")
        print(f"Cs_Pb_ratio (old): {Cs_Pb_ratio}")
        print(f"KRR predicted peak position:    {self.NPL_model.predict(input_NPL)}\n")
        print("----------------------------------------------------")

        print("Do you want to print the results? (y/n)")
        if input() == "y":

            print("writing to file ...")

            # write to file
            with open("suggestions.txt", "a") as file:

                file.write(f"\n")
                file.write(f"----------------------------------------------------   {str(date.today())}   ----------------------------------------------------\n")
                file.write(f"L-SYNTH 2.0-{np.random.randint(10000)}\n")
                file.write(f"THE SYNTHESIZER RECOMMENDS: \n")
                file.write(f"\n")
                file.write(f"When using -{self.molecule}- as an antisolvent \n")
                file.write(f"to make NPLs around {self.peak}nm:\n")
                file.write(f"Choose the following synthesis parameters:\n")
                file.write(f"{results_string}\n")

                file.write(f"with the following As/Pb ratio: {As_Pb_ratio*10}\n")
                file.write(f" and the following Cs/Pb ratio: {Cs_Pb_ratio}\n")

                if "PLQY" in self.obj:
                    file.write(f"GP predicted rel. PLQY for this sample:   +{self.PLQY_model.predict([input_PLQY])}\n")

                file.write(f"GP predicted peak position: {self.NPL_model.predict(input_NPL)}\n")

                if "FWHM" in self.obj:
                    file.write(f"GP predicted rel. FWHM: {self.FWHM_model.predict([input_FWHM])}\n")

                file.write(f"\n")
                file.write(f"[ DEV --> Opt. Iterations: {self.iterations}, fixed c_Pb: {self.c_Pb_fixed}, fixed V_As: {self.V_As_fixed}, Cs_Pb_opt: {self.Cs_Pb_opt}, Cs_As_opt: {self.Cs_As_opt}\n")
                file.write(f"model type: {self.model_type}, encoding: {self.encoding_type}, geometry: {self.geometry}, c_Pb_max: {self.c_Pb_max} ]\n")
                file.write(f"\n")
                file.write(f"THANK YOU FOR USING THE SYNTHESIZER MODULE!\n")
            
            print("done writing to file ...")


    def test_results(self, x):

        """ Test the results against a different Gaussian Process model 
        (ideally with different hyperparameters)
        """

        print("testing results...")

        gp_PLQY = GaussianProcess(training_data = np.array(self.inputs_PLQY),
                            parameter_selection = self.parameters_PLQY,
                            targets = np.array(self.PLQY), 
                            kernel_type = "LIN",
                            model_type  = "GPRegression",
                            )

        gp_PLQY.train()

        print(f"GP predicted PLQY for this sample:   {gp_PLQY.predict(x)}")
    

    def print_logo(self):
        """
            Print the logo to the console
            -> ASCII image from ascii-art-generator.org
        """

        # read in from file (ascii-art.txt)
        with open("ascii-art.txt", "r") as file:
            for line in file:
                print(line, end="")
                time.sleep(0.02)	

        print("\n")
        