""" Gaussian Process Regression with GPy"""
    # < github.com/leoluber >


import numpy as np
import matplotlib.pyplot as plt
import GPy
from timeit import default_timer as timer
from typing import Literal

# custom
from Datastructure import *
from helpers import *




class GaussianProcess:
    
    """ Gaussian Process Regression

    A Gaussian Process model for regression tasks with the GPy library and a set of
    helper functions for hyperparameter optimization and model evaluation

    BASICS
    ------
    - uses GPy
    - optimized hyperparameters
    - can optimize the choice of input parameters

    PARAMETERS
    ----------
    - training_data: list of training data (list)
    - targets: list of targets (list)
    - parameter_selection: list of parameter names (list)
    - kernel_type: type of kernel (str)
    - model_type: type of model (str, fixed)

    USAGE
    -----

    >>> gp = GaussianProcess(training_data, targets, parameter_selection, ...)
    >>> gp.leave_one_out_cross_validation(training_data, targets)
    >>> gp.train()
    >>> gp.predict(sample)

    """


    def __init__(self, 
                 training_data, parameter_selection = None,
                 targets = None,
                 kernel_type: Literal["RBF", "MLP", "EXP", "LIN"] = "RBF",
                 model_type  = "GPRegression",
                 ):
        

        # training specific
        self.training_data = training_data
        self.input_dim = self.training_data.shape[1]
        self.parameter_selection = parameter_selection
        self.targets = targets


        # model specific
        self.kernel_type = kernel_type
        self.kernel = self.kernel = self.get_kernel(self.kernel_type, input_dim = self.input_dim)   
        self.model_type = model_type
        self.model = None


        # evaluation (LOO) results
        self.loo_predictions, self.loo_uncertainty, self.loo_error = None, None, None

        # performance
        self.fitting_time = None


### ----------------------- BASICS ------------------------- ###


    def train(self) -> GPy.models.GPRegression:
        
        """Trains the Gaussian Process model """


        # adjust dimensions
        self.targets = np.reshape(self.targets, (self.targets.shape[0], 1))
        self.training_data = np.reshape(self.training_data, (self.training_data.shape[0], self.training_data.shape[1]))     

        # select the model
        if self.model_type == "GPRegression":
            model = GPy.models.GPRegression(self.training_data, self.targets, self.kernel)
        else:
            raise ValueError("Model type not supported")

        model.optimize()

        self.model = model
        return model	


    def predict(self, sample) -> np.ndarray:

        """ Predicts the target value for a given sample"""

        sample = np.reshape(sample, (1, len(sample)))
        return self.model.predict(sample)



### ------------------ CROSS VALIDATION -------------------- ###

    def leave_one_out_cross_validation(self, training_data, targets) -> float:

        """LOO cross validation on the training data, returns the mean squared error"""


        # LOO loop
        predictions, uncertainty, error = [], [], []

        for i in range(len(training_data)):

            print("step: "+ str(i) + "  / " + str(len(training_data)-1))

            # Split the data into training and test data and reshape so it fits the GPy standard
            X_train = np.delete(training_data, i, axis=0)
            y_train = np.delete(targets, i, axis=0)

            # reshaping for GPy standard
            y_train = np.reshape(y_train, (y_train.shape[0], 1))
            X_test = np.reshape(training_data[i], (1, training_data[i].shape[0]))
            y_test = np.reshape(targets[i], (1, 1))

            # select the model
            self.kernel = self.get_kernel(self.kernel_type, input_dim = X_train.shape[1])

            if self.model_type == "GPRegression":
                model = GPy.models.GPRegression(X_train, y_train, self.kernel)
            else:
                raise ValueError("Model type not supported")


            # opimize the model and predict the test data
            if self.fitting_time is None:
                start = timer()

            model.optimize()

            if self.fitting_time is None:
                end = timer()
                self.fitting_time = end - start

            # predict the test data
            mean, variance = model.predict(X_test)

            # store the results
            predictions.append(mean[0][0])
            uncertainty.append(variance[0][0])
            error.append(np.abs(mean[0][0] - y_test[0][0]))
        

        self.loo_predictions, self.loo_uncertainty, self.loo_error = np.array(predictions), np.array(uncertainty), np.array(error)

        print(f"Mean squared error: {np.mean(self.loo_error)}")
        print(f"Fitting time: {self.fitting_time}")

        return np.mean(self.loo_error)
    


### --------------------- OPTIMIZATION ----------------------- ###

    def iterate_and_ignore(self):
        
        """ Iterates over all parameters and ignores one at a time to find the best set """

        mse_historgram = []

        for parameter in self.parameter_selection:

            print(f"Ignoring parameter:     {parameter}")

            index = self.parameter_selection.index(parameter)
            new_training_data = np.delete(self.training_data, index, axis=1)

            mse = self.leave_one_out_cross_validation(new_training_data, self.targets)
            mse_historgram.append(mse)
        

        # plot the results
        fig = plt.figure(figsize = (10, 5))    
        plt.bar(self.parameter_selection,  mse_historgram, width = 0.4)

        plt.xlabel("Parameters")
        plt.ylabel("Error(MSE)")

        plt.show()
    

    def choose_best_kernel(self):
        
        """ Iterates over all kernels and chooses the best one """


        mse_historgram = []
        for kernel in ["RBF", "MLP", "EXP", "LIN"]:

            print(f"Testing kernel:     {kernel}")

            self.kernel = self.get_kernel(kernel, input_dim = self.input_dim)

            mse = self.leave_one_out_cross_validation(self.training_data, self.targets)
            mse_historgram.append(mse)
        
        self.kernel_type = ["RBF", "MLP", "EXP", "LIN"][np.argmin(mse_historgram)]

        # plot the results
        fig = plt.figure(figsize = (10, 5))    
        plt.bar(["RBF", "MLP", "EXP", "LIN"],  mse_historgram, width = 0.4)

        plt.xlabel("Kernels")
        plt.ylabel("Error(MSE)")

        plt.show()



### ------------- HELPER FUNCTIONS --------------- ###

    def get_kernel(self, kernel, input_dim = None) -> GPy.kern:

        """ Returns the kernel object based on the kernel string """

        match kernel:
            case "RBF":
                return GPy.kern.RBF(input_dim)
            case "MLP":
                return GPy.kern.MLP(input_dim, ARD = True) + GPy.kern.MLP(input_dim, ARD=True)
            case "EXP":
                return GPy.kern.Exponential(input_dim, ARD = True)
            case "LIN":
                return GPy.kern.Linear(input_dim)
            case "POLY":
                return GPy.kern.Poly(input_dim)
            case _:
                raise ValueError("Kernel not supported")
            


### ------------- PLOTTING FUNCTIONS --------------- ###

    def regression_plot(self):

        """ Plots the regression results """

        plt.figure(figsize=(10, 6))
        plt.scatter(self.targets, self.loo_predictions, c='r', label='Predictions')
        plt.plot([self.targets.min(), self.targets.max()], [self.targets.min(), self.targets.max()], 'k--', lw=2)

        plt.legend()
        plt.show()
