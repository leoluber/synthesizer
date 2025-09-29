""" 
    Project:     synthesizer
    File:        helpers.py
    Description: Helper functions for the synthesizer project
    Author:      << github.com/leoluber >> 
    License:     MIT
"""



import numpy as np
import matplotlib.pyplot as plt
import GPy
from typing import Literal


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
    OR:
    >>> gp = GaussianProcess(training_data, targets, parameter_selection, ...)
    >>> gp.train()
    >>> gp.predict(sample)

    where sample has to be of the form ... TODO

    """


    def __init__(self, 
                 training_data, 
                 targets,
                 kernel_type: Literal["RBF", "EXP", "LIN",] = "EXP",
                 ):
        
        # training specific
        self.training_data = training_data
        self.input_dim = self.training_data.shape[1]
        self.targets = targets

        # model specific
        self.kernel_type = kernel_type
        self.kernel = self.get_kernel(self.kernel_type, input_dim = self.input_dim)   
        self.model = None

        # evaluation (LOO) results
        self.loo_predictions, self.loo_uncertainty, self.loo_error = None, None, None


### ----------------------- BASICS ------------------------- ###

    def train(self,
              training_data = None, targets = None,
              ) -> GPy.models.GPRegression:
        
        """ Trains the Gaussian Process model 
        
        ARGS (optional)
        ---------------
        - training_data: training data (np.array)
        - targets: targets (np.array)
        [if not provided, the class variables are used]

        RETURNS
        -------
        - model: GPy.models.GPRegression object

        """
        print("Training the model...")

        # adjust dimensions
        if training_data is None or targets is None:
            targets = np.reshape(self.targets, (self.targets.shape[0], 1))
            training_data = np.reshape(self.training_data, 
                                       (self.training_data.shape[0], self.training_data.shape[1]))

        else:
            targets = np.reshape(targets, (targets.shape[0], 1))
            training_data = np.reshape(training_data, 
                                       (training_data.shape[0], training_data.shape[1]))

        # initialize the model and optimize
        model = GPy.models.GPRegression(training_data, targets, self.kernel)
        model.optimize()
        self.model = model

        return model


    def predict(self, sample) -> np.ndarray:

        """ Predicts the target value for a given sample
        
        ARGS
        ----
        - sample: input sample (np.array)

        RETURNS
        -------
        - prediction: predicted target value (np.array)
        """

        sample = np.reshape(sample, (1, len(sample)))
        return self.model.predict(sample)


    def get_kernel(self, kernel, input_dim) -> GPy.kern:

        """ Returns the kernel object based on the kernel string 
        
        ARGS
        ----
        - kernel: kernel string (str)
        - input_dim: input dimensions (int)

        RETURNS
        -------
        - kernel: GPy.kern object

        """

        match kernel:
            case "RBF":
                return GPy.kern.RBF(input_dim, lengthscale= 0.1)
            case "EXP":
                return GPy.kern.Exponential(input_dim, lengthscale= 0.15)
            case "LIN":
                return GPy.kern.Linear(input_dim)
            case _:
                raise ValueError("Kernel not supported")
            

### ------------------ CROSS VALIDATION -------------------- ###

    def leave_one_out_cross_validation(self, training_data, targets, 
                                       baseline_list = None, include_sample = None,) -> float:


        """LOO cross validation on the training data, returns the mean squared error"""

        # LOO loop
        predictions, uncertainty, error = [], [], []

        # default values
        if include_sample is None:
            include_sample = np.ones(len(training_data))

        if baseline_list is None:
            baseline_list = np.zeros(len(training_data))

        step = 0
        for i, data in enumerate(training_data):

            step += 1

            # critically the baseline has to be excluded from the LOO
            if baseline_list[i] == True:
                continue
            if include_sample[i] == False:
                continue

            print("step: "+ str(step) + "  / " + str(len([inc for inc in include_sample if inc == True])))

            # Split the data into training and test data
            X_train = np.delete(training_data, i, axis=0)
            y_train = np.delete(targets, i, axis=0)

            # reshaping for GPy standard
            y_train = np.reshape(y_train, (y_train.shape[0], 1))
            X_test  = np.reshape(training_data[i], (1, training_data[i].shape[0]))
            y_test  = np.reshape(targets[i], (1, 1))

            # select the model
            self.kernel = self.get_kernel(self.kernel_type, input_dim = X_train.shape[1])
            model = GPy.models.GPRegression(X_train, y_train, self.kernel)

            model.optimize()

            # predict the test data
            mean, variance = model.predict(X_test)

            # store the results
            predictions.append(mean[0][0])
            uncertainty.append(variance[0][0])
            error.append(np.abs(mean[0][0] - y_test[0][0]))
        

        self.loo_predictions, self.loo_uncertainty, self.loo_error = np.array(predictions), np.array(uncertainty), np.array(error)

        print(f"Mean squared error: {np.mean(self.loo_error)}")
        print(f"Median squared error: {np.median(self.loo_error)}")

        # decide between mean and median error
        #return np.mean(self.loo_error)
        return np.median(self.loo_error)
    

    def molecular_cross_validation(self, data_objects, 
                                   transfer_molecule,) -> float:

        """
        Transfer cross validation for a specific molecule
        """

        # training
        x_train = [data["encoding"] + data["total_parameters"] for data in data_objects 
                   if data["molecule_name"] != transfer_molecule]
        y_train = [data["y"] for data in data_objects 
                   if data["molecule_name"] != transfer_molecule]

        # testing (we dont want to test against the baseline, as it is not a real molecule)
        x_test = [data["encoding"] + data["total_parameters"] for data in data_objects 
                  if data["molecule_name"] == transfer_molecule and data["baseline"] == False]
        y_test = [data["y"] for data in data_objects 
                  if data["molecule_name"] == transfer_molecule and data["baseline"] == False]

        # reshape for GPy
        y_train = np.reshape(np.array(y_train), (np.array(y_train).shape[0], 1))
        y_test  = np.reshape(np.array(y_test), (np.array(y_test).shape[0], 1))
        x_train = np.reshape(np.array(x_train), (np.array(x_train).shape[0], np.array(x_train).shape[1]))
        x_test  = np.reshape(np.array(x_test), (np.array(x_test).shape[0], np.array(x_test).shape[1]))

        # select the kernel
        self.kernel = self.get_kernel(self.kernel_type, input_dim = x_train.shape[1])

        # train the model
        model = GPy.models.GPRegression(x_train, y_train, self.kernel)
        model.optimize()
        predictions, _ = model.predict(x_test)

        # calculate the error
        mean_error = np.mean(np.abs(predictions - y_test))
        median_error = np.median(np.abs(predictions - y_test))

        self.loo_predictions = predictions
        self.targets = y_test

        return median_error
        # return mean_error


### ------------------ PLOTTING FUNCTIONS ------------------ ###

    def regression_plot(self, TRANSFER_MOLECULE = None):

        """ Plots the regression results 
        """

        if self.loo_predictions is None:
            raise ValueError("No LOO predictions available. Please run leave_one_out_cross_validation() first.")

        fig, ax = plt.subplots(figsize = (4, 4))

        error        = np.mean(np.abs(np.array(self.targets) - np.array(self.loo_predictions)))
        error        = round(error, 4)
        error_median = np.median(np.abs(np.array(self.targets) - np.array(self.loo_predictions)))
        error_median = round(error_median, 4)

        ax.scatter(self.targets, self.loo_predictions, c='blue', label= f'med err: {error_median}', marker = 's',s = 40)
        
        # plot the identity line
        ax.plot([min(self.targets), max(self.targets)], [min(self.targets), max(self.targets)], 'k--', lw=2)


        # plot settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12)
        ax.xaxis.set_label_text("true value", fontsize = 12)
        ax.yaxis.set_label_text("predicted value", fontsize = 12)

        plt.tight_layout()
        plt.legend()
        plt.show()

    def print_parameters(self):

        """ Prints the parameters of the model """

        print(self.model)