import numpy as np
from GPyOpt.methods import BayesianOptimization
from datetime import date

# custom modules
from Datastructure import *
from helpers import *
from KRR import Ridge
from GaussianProcess import GaussianProcess


"""
    Optimization Module with Synthesis Parameter Recommendations for Perovskite NPL synthesis;
    Samples synthesis parameters for a given NPL type while optimizing the target value (PLQY, ...)
    - models:           Kernel Ridge Regression
    - datastructure:    DatastructureGP (custom)
    - optimization:     GPyOpt (Bayesian Optimization)
"""


class Synthesizer:
    def __init__(self,
                 molecule : str,
                 iterations : int,
                 peak : int,
                 obj : str = ["PEAK_POS", "PLQY", "FWHM"]
                ):
        
            # set of parameters used in the objective function
        self.obj = obj

            # main initialization
        self.datastructure_NPL, self.datastructure_PLQY, self.datastructure_FWHM= self.get_datastructures()
        self.iterations = iterations
        self.molecule_names = self.datastructure_NPL.molecule_names
        self.one_hot_molecule = self.one_hot_encode_molecule(molecule)

            # target peak position
        self.peak = peak

            # data
        self.inputs_NPL, self.peak_pos =        self.get_data(self.datastructure_NPL)
        self.inputs_PLQY, self.PLQY   =         self.get_data(self.datastructure_PLQY)
        if "FWHM" in self.obj:
            self.inputs_FWHM, self.FWHM   =     self.get_data(self.datastructure_FWHM)
        self.parameters_NPL  =                  self.datastructure_NPL.synthesis_training_selection
        self.parameters_PLQY =                  self.datastructure_PLQY.synthesis_training_selection
        if "FWHM" in self.obj:  
            self.parameters_FWHM =              self.datastructure_FWHM.synthesis_training_selection

            # models
        self.NPL_model  = self.train(self.inputs_NPL, self.peak_pos, self.parameters_NPL)
        self.PLQY_model = self.train(self.inputs_PLQY, self.PLQY, self.parameters_PLQY)
        if "FWHM" in self.obj:
            self.FWHM_model = self.train(self.inputs_FWHM, self.FWHM, self.parameters_FWHM)

            # sampling
        self.limits = self.get_limits()

            # results
        self.results = None



### ------------------------ SAMPLING ------------------------ ###

    def optimize_NPL(self):
        """
            Optimize for the selected NPL type using the GPyOpt library
            - minimizes the distance to the perfect peak position
            - returns the optimal parameters and the best peak position
        """
        print("optimizing NPL ...")


        # define the optimization domain from the limits dictionary
        bounds = [{'name': parameter, 
                   'type': 'continuous', 
                   'domain': (self.limits[parameter][0], 
                              self.limits[parameter][1])} 
                              for parameter in self.parameters_PLQY[:]]


        # define the objective function
        def f(x):
            """
                The As_Pb ratio is calculated from the input parameters since it is not included in the optimization itself
                - it is used here as an input for the NPL model (#MLs)
            """

            # calculate As_Pb ratio
            As_Pb_ratio = x[0][0] / ((x[0][1]*x[0][2]))
            max_As_Pb_ratio = self.datastructure_NPL.max_min["AS_Pb_ratio"][0]
            min_As_Pb_ratio = self.datastructure_NPL.max_min["AS_Pb_ratio"][1]
            As_Pb_ratio = As_Pb_ratio / max_As_Pb_ratio                                                # -------> normalize As_Pb ratio, might change to different method !!!!!!!!


            # add the one hot encoded molecule to both inputs
            input =           np.array(self.one_hot_molecule)
            PLQY_input =      np.append(input, x)
            PLQY_input =      [value for value in PLQY_input]


            NPL_input =       np.append(input, As_Pb_ratio)
            FWHM_input =      np.append(NPL_input, x) 
            NPL_input =       [value for value in NPL_input]
            FWHM_input =      [value for value in FWHM_input]
        
            # return the loss
            PLQY =  self.PLQY_model.predict([PLQY_input])
            NPL =   self.NPL_model.predict([NPL_input])
            if "FWHM" in self.obj:
                FWHM =  self.FWHM_model.predict([FWHM_input])

            output = (1 - PLQY)**2  + ((NPL - self.peak)**2)             # minimize the distance to the perfect peak position
            # output = (1 - PLQY)**2  * ((NPL - self.peak)**2)           # same but different weighting
            if "FWHM" in self.obj:
                output += FWHM

            return output


        # optimize
        optimizer = BayesianOptimization(f = f, domain = bounds)
        optimizer.run_optimization(max_iter = self.iterations)

        self.results = self.return_results(optimizer.x_opt)
        return optimizer.x_opt, optimizer.fx_opt



### -------------------------- INIT -------------------------- ###
    
    def train(self, inputs, targets, parameter_selection):
        """
            Train a RKK model on the given data
        """
        rkk = Ridge(inputs, targets, parameter_selection, kernel_type= "polynomial", alpha= 1e-1, gamma= 0.01)

        # optimize hyperparameters
        print("finding hyperparameters ... ")
        rkk.optimize_hyperparameters()

        rkk.fit()
        return rkk
    
    def train_gp(self, inputs, targets, parameter_selection):
        """
            Train a RKK model on the given data
        """
        rkk = Ridge(inputs, targets, parameter_selection, kernel_type= "polynomial", alpha= 1e-1, gamma= 0.01)

        # optimize hyperparameters
        print("finding hyperparameters ... ")
        rkk.optimize_hyperparameters()

        rkk.fit()
        return rkk

    def get_data(self, datastructure : Datastructure):
        """
            Get data from the Datastructure object
        """
        data_objects = datastructure.get_data()

        inputs =  [data.one_hot_molecule + data.total_parameters for data in data_objects]     # including one hot encoding
        target  = [data.y for data in data_objects]

        return inputs, target


    def get_limits(self):
        """
            Defines the limits for the NPL synthesis parameters from the peak range,
            returns them as a dictionary
        """
        limits = {}
        samples = [input[15:] for input in self.inputs_PLQY]                                   # exclude the one hot encoded molecule
        
        for i, parameter in enumerate(self.parameters_PLQY):
            limits[parameter] = [min([sample[i] for sample in samples]), 
                                 max([sample[i] for sample in samples])]
        
        # increase the values for the lead bromide concentration
        limits["c (PbBr2)"] = [0.08,
                               limits["c (PbBr2)"][1] * 1.5]

        return limits
    

    def get_discrete_values(self, parameter):
        """
            Get all the discrete values for the specified parameter
        """
        index = self.parameters_NPL.index(parameter)
        return list(set([sample[index] for sample in self.inputs_NPL]))
    

    def one_hot_encode_molecule(self, molecule):
        """
            One hot encode the value
        """
        one_hot_molecule = [0] * len(self.molecule_names)
        one_hot_molecule[self.molecule_names.index(molecule)] = 1
        return one_hot_molecule


    def get_datastructures(self):
        """
            Get the datastructures for the NPL and PLQY models
            --> make adjustments here if necessary
        """
        datastructure_NPL = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                                        target = "PEAK_POS",                                                   # "PLQY", "FWHM", "PEAK_POS"
                                        wavelength_unit= "NM",                                                 # "NM", "EV"
                                        exclude_no_star= False,
                                        wavelength_filter= [400, 550],
                                        )
        datastructure_NPL.synthesis_training_selection = ["AS_Pb_ratio",] 


        datastructure_PLQY = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv",
                                        target = "PLQY",                                                       # "PLQY", "FWHM", "PEAK_POS"
                                        wavelength_unit= "NM",                                                 # "NM", "EV"
                                        exclude_no_star= True,
                                        wavelength_filter= [440, 510],
                                        )
        datastructure_PLQY.synthesis_training_selection = ["V (antisolvent)", "c (PbBr2)", "V (PbBr2 prec.)", "V (Cs-OA)" ,]


        datastructure_FWHM = Datastructure(synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv",
                                        target = "FWHM",                                                       # "PLQY", "FWHM", "PEAK_POS"
                                        wavelength_unit= "NM",                                                 # "NM", "EV"
                                        exclude_no_star= False,
                                        wavelength_filter= [400, 510],
                                        monodispersity_only= True,
                                        )
        datastructure_FWHM.synthesis_training_selection = ["AS_Pb_ratio", "V (antisolvent)", "c (PbBr2)", "V (PbBr2 prec.)", "V (Cs-OA)" ,]
    
        return datastructure_NPL, datastructure_PLQY, datastructure_FWHM
            


### ------------------------ PRINTING AND TESTING ------------------------ ###

    def return_results(self, x):
        """
            return the results from the optimization 
        """

        results = {}

        # get As/Pb ratio
        max_As_Pb_ratio =  self.datastructure_NPL.max_min["AS_Pb_ratio"][0]
        results["As_Pb_ratio"] =      x[0] / (x[1] * x[2] * max_As_Pb_ratio)

        # get initial input that leads to the best plqy value
        one_hot =                   np.array(self.one_hot_molecule)
        input_NPL =                 np.append(one_hot, results["As_Pb_ratio"])
        input_PLQY =                np.append(one_hot, x)
        results["input_NPL"] =      [value for value in input_NPL]
        results["input_PLQY"] =     [value for value in input_PLQY]


        # denormalize the parameters
        results["results_string"] = []
        denorm = {}
        for i, parameter in enumerate(self.parameters_PLQY):
            denorm[parameter] = self.datastructure_PLQY.denormalize(x[i], parameter)
            results["results_string"].append(f"{parameter} :  {denorm[parameter]}")
        results["As_Pb_ratio_denorm"] = denorm["V (antisolvent)"] / (denorm["c (PbBr2)"] * denorm["V (PbBr2 prec.)"])

        return results



    def print_results(self, results_string, x, x_peak_pos, As_Pb_ratio):
        """
            Print the results to a file
        """
        # write to file
        with open("suggestions.txt", "a") as file:
            file.write(f"\n")
            file.write(f"----------------------------------------------------   {str(date.today())}   ----------------------------------------------------\n")
            file.write(f"L-SYNTH-{np.random.randint(1000)}\n")
            file.write(f"THE SYNTHESIZER RECOMMENDS: \n")
            file.write(f"\n")
            file.write(f"When using -{self.molecule_names[self.one_hot_molecule.index(1)]}- as an antisolvent \n")
            file.write(f"to make NPLs around {self.peak}nm:\n")
            file.write(f"Choose the following synthesis parameters:\n")
            file.write(f"{results_string}\n")
            file.write(f"with the following As/Pb ratio: {As_Pb_ratio}\n")
            file.write(f"KRR predicted PLQY for this sample:   {self.PLQY_model.predict([x])}\n")
            file.write(f"KRR predicted peak position:   {self.NPL_model.predict([x_peak_pos])}\n")
            file.write(f"\n")
            file.write(f"[ DEV -->   KRR_PLQY range: {self.datastructure_PLQY.wavelength_filter},  Opt. Iterations: {self.iterations}]\n")
            file.write(f"\n")
            file.write(f"THANK YOU FOR USING THE SYNTHESIZER MODULE!\n")


    def test_results(self, x):
        # test the results with a separate Gaussian Process
        print("testing results...")

        gp_PLQY = GaussianProcess(training_data = np.array(self.inputs_PLQY),
                            parameter_selection = self.parameters_PLQY,              # for running "iterate_and_ignore()"
                            targets = np.array(self.PLQY), 
                            kernel_type = "LIN",                                    # "RBF", "MLP", "EXP", "LIN"
                            model_type  = "GPRegression",
                            )

        
        gp_PLQY.train()
        print(f"GP predicted PLQY for this sample:   {gp_PLQY.predict(x)}")


    def plot_suggestions(self, opt_x, datastructure, parameters):
        """
            Plot the prediction in relation to all relevant data points with correct wavelength filter and molecule
        """
        
        fig, ax = datastructure.plot_data(parameters[0], parameters[1], parameters[2], parameters)                  # plot data distribution
        ax.scatter(opt_x[0], opt_x[1], opt_x[2], color='red', marker="x", s=200)
        #plt.show()
        return fig, ax
        