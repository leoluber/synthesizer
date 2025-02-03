""" Gaussian Process Regression with GPy"""
    # < github.com/leoluber >


import numpy as np
import matplotlib.pyplot as plt
import GPy
from timeit import default_timer as timer
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
    >>> gp.train()
    >>> gp.predict(sample)

    """


    def __init__(self, 
                 training_data = None, 
                 targets = None,
                 kernel_type: Literal["RBF", "MLP", "EXP", "LIN"] = "EXP",
                 model_type  = "GPRegression",
                 ):
        

        # training specific
        self.training_data = training_data
        self.input_dim = self.training_data.shape[1]
        self.targets = targets


        # model specific
        self.kernel_type = kernel_type
        self.kernel = self.get_kernel(self.kernel_type, input_dim = self.input_dim)   
        self.model_type = model_type
        self.model = None


        # evaluation (LOO) results
        self.loo_predictions, self.loo_uncertainty, self.loo_error = None, None, None

        # performance
        self.fitting_time = None


### ----------------------- BASICS ------------------------- ###


    def train(self,
              training_data = None, targets = None,
              ) -> GPy.models.GPRegression:
        
        """Trains the Gaussian Process model """


        # adjust dimensions
        if training_data is None or targets is None:
            targets = np.reshape(self.targets, (self.targets.shape[0], 1))
            training_data = np.reshape(self.training_data, (self.training_data.shape[0], self.training_data.shape[1]))

        else:
            targets = np.reshape(targets, (targets.shape[0], 1))
            training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1]))

        # select the model
        if self.model_type == "GPRegression":
            model = GPy.models.GPRegression(training_data, targets, self.kernel)
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


    def leave_one_out_cross_validation(self, training_data, targets, 
                                       baseline_list = None, include_sample = None,) -> float:


        """LOO cross validation on the training data, returns the mean squared error"""


        # LOO loop
        predictions, uncertainty, error = [], [], []

        if include_sample is None:
            include_sample = np.ones(len(training_data))

        step = 0

        for i, data in enumerate(training_data):

            step += 1

            if baseline_list is not None and include_sample is not None:
                # we dont want to test against the baseline, as it is not a real molecule
                if baseline_list[i] == True:
                    continue
                
                # we only want to test against the samples that are "included"
                if include_sample[i] == False:
                    continue

            print("step: "+ str(step) + "  / " + str(len([inc for inc in include_sample if inc == True])))

            # Split the data into training and test data and reshape so it fits the GPy standard
            X_train = np.delete(training_data, i, axis=0)
            y_train = np.delete(targets, i, axis=0)

            # reshaping for GPy standard
            y_train = np.reshape(y_train, (y_train.shape[0], 1))
            X_test  = np.reshape(training_data[i], (1, training_data[i].shape[0]))
            y_test  = np.reshape(targets[i], (1, 1))

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

        if baseline_list is not None and include_sample is not None:
            self.targets = np.array([t for i, t in enumerate(targets) if baseline_list[i] == False and include_sample[i] == True])

        else:
            self.targets = np.array([t for i, t in enumerate(targets)])

        print(f"Mean squared error: {np.mean(self.loo_error)}")
        print(f"Median squared error: {np.median(self.loo_error)}")
        print(f"Fitting time: {self.fitting_time}")

        # regres

        #return np.mean(self.loo_error)
        return np.median(self.loo_error)
    


    def molecular_cross_validation(self, data_objects, 
                                   transfer_molecule,) -> float:

        """
        Cross validation on the training data for a specific molecule
        """

        # training
        x_train = [data["encoding"] + data["total_parameters"] for data in data_objects if data["molecule_name"] != transfer_molecule]
        y_train = [data["y"] for data in data_objects if data["molecule_name"] != transfer_molecule]

        # testing (we dont want to test against the baseline, as it is not a real molecule)
        x_test = [data["encoding"] + data["total_parameters"] for data in data_objects if data["molecule_name"] == transfer_molecule and data["baseline"] == False]
        y_test = [data["y"] for data in data_objects if data["molecule_name"] == transfer_molecule and data["baseline"] == False]

        # reshape
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # reshape for GPy
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))


        # select the kernel
        self.kernel = self.get_kernel(self.kernel_type, input_dim = x_train.shape[1])

        # train the model
        model = GPy.models.GPRegression(x_train, y_train, self.kernel)
        model.optimize()

        # predict the test data
        predictions, _ = model.predict(x_test)

        # calculate the error
        #error = np.mean(np.abs(predictions - y_test))
        error = np.median(np.abs(predictions - y_test))
        print(f"Median error: {error}")

        # regression plot
        self.loo_predictions = predictions
        self.targets = y_test
        #self.regression_plot()

        return error



    def active_learning_simulation(self, data_objects,
                                   measured_molecule: str = "Methanol",
                                   resolution = 5,	
                                   ) -> float:


        """ Active learning simulation 
        
        - samples from the data to train a new model from scratch
        - tests against the fully trained model

        TODO: implement different versions of this
        
        """

        # seperate into baseline and real molecules
        starting_data = [data["encoding"] + data["total_parameters"]
                         for data in data_objects if data["baseline"] == True or data["molecule_name"] != measured_molecule]
        self.targets = np.array([data["y"] for data in data_objects if data["baseline"] == True or data["molecule_name"] != measured_molecule])

        measured_data = [data["encoding"] + data["total_parameters"]
                         for data in data_objects if data["baseline"] == False and data["molecule_name"] == measured_molecule]
        measured_targets = np.array([data["y"] for data in data_objects if data["baseline"] == False and data["molecule_name"] == measured_molecule])
        
        #shuffle 
        zip_list = list(zip(measured_data, measured_targets))
        np.random.shuffle(zip_list)
        measured_data, measured_targets = zip(*zip_list)


        # train the model on the starting data
        self.training_data = np.array(starting_data)
        self.targets = np.array(self.targets)
        self.train()


        # main loop
        err_matrix = np.zeros((len(measured_data), len(measured_data)))
        for n in range(0, len(measured_data), resolution):

            """ iterate over the measured data and define the test data """


            print(f"Step: {n} / {int(len(measured_data))}")


            # get the test data
            test_data    = measured_data[n:n+resolution]
            test_targets = measured_targets[n:n+resolution]
            

            # break if the test data is too small
            if len(test_data) < resolution:
                break


            # remaining data
            if n!=0:
                remaining_data = np.vstack((measured_data[:n], measured_data[n+resolution:]))
                remaining_targets = np.append(measured_targets[:n], measured_targets[n+resolution:])
            else:
                remaining_data = measured_data[resolution:]
                remaining_targets = measured_targets[resolution:]


            for l in range(0, len(remaining_data), resolution):

                """ iterate over different numbers of experiments """
            
                print(f"Step: {l} / {int(len(measured_data))}")


                # add the experiment to the training data
                if l!=0:
                    training_data = np.vstack((self.training_data, remaining_data[:l]))
                    targets = np.append(self.targets, remaining_targets[:l])
                else:
                    training_data = self.training_data
                    targets = self.targets

                # train the model on the new data
                self.train(training_data = training_data, targets = targets)


                # predict the experiment
                predictions =[self.predict(sample)[0][0][0] for sample in test_data]
                error = np.mean(np.abs(np.array(predictions) - np.array(test_targets)))
                #error = np.median(np.abs(np.array(predictions) - np.array(test_targets)))

                # append to lists
                err_matrix[int(n/resolution), int(l/resolution)] = error

                print(f"Error: {error}")



        # plot the results
        err_matrix = err_matrix[:4, :4]
        print(err_matrix)
        error = [np.mean(err_matrix[:, n]) for n in range(len(err_matrix))]
        num_exp = [i*resolution for i in range(len(err_matrix))]
        self.plot_error(error, num_exp, molecule = measured_molecule)

        return self.model









### --------------------- OPTIMIZATION ----------------------- ###


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
                return GPy.kern.RBF(input_dim, lengthscale= 0.1)
            case "MLP":
                return GPy.kern.MLP(input_dim, ARD = True) + GPy.kern.MLP(input_dim, ARD=True)
            case "EXP":
                return GPy.kern.Exponential(input_dim, lengthscale= 0.5)
            case "LIN":
                return GPy.kern.Linear(input_dim)
            case "POLY":
                return GPy.kern.Poly(input_dim, order = 3,)
            case "SPECIAL":
                parameter_kernel = GPy.kern.Exponential(2, active_dims= [5,6])
                parameter_kernel.lengthscale.constrain_bounded(100, 600)
                parameter_kernel.variance.constrain_bounded(100, 600)
                geometry_kernel = GPy.kern.Exponential(5, active_dims=[0,1,2,3,4,],)
                geometry_kernel.lengthscale.constrain_bounded(10, 200)
                geometry_kernel.variance.constrain_bounded(0.1, 200)

                return parameter_kernel * geometry_kernel 
                #return exp * rbf
            case _:
                raise ValueError("Kernel not supported")
            


### ------------- PLOTTING FUNCTIONS --------------- ###

    def regression_plot(self, TRANSFER_MOLECULE = None):

        """ Plots the regression results """

        fig, ax = plt.subplots(figsize = (4.2, 4))

        print(len(self.targets), len(self.loo_predictions))

        error = np.mean(np.abs(np.array(self.targets) - np.array(self.loo_predictions)))
        error_median = np.median(np.abs(np.array(self.targets) - np.array(self.loo_predictions)))
        error_median = round(error_median, 4)
        error = round(error, 4)
        ax.scatter(self.targets, self.loo_predictions, c='blue', label= f'med err: {error_median}', marker = 's',s = 40)
        
        if TRANSFER_MOLECULE is None:
            #ax.plot([0.06, 0.18],[0.06, 0.18] ,'k--', lw=2)
            #ax.plot([0.0, 1],[0.0, 1] ,'k--', lw=2)
            #ax.plot([440, 520], [440, 520], 'k--', lw=2)
            ax.plot([570, 710],[570, 710] ,'k--', lw=2)
        else:
            ax.plot([420, 520], [420, 520], 'k--', lw=2)


        # settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12)

        ax.xaxis.set_label_text("True value", fontsize = 12)
        ax.yaxis.set_label_text("Predicted value", fontsize = 12)

        #ax.set_xlim(430, 530)
        #ax.set_ylim(430, 530)

        plt.tight_layout()
        plt.legend()
        plt.show()

        # save as svg
        #fig.savefig(f"regression_plot_{TRANSFER_MOLECULE}_err_med_{error_median}_.svg", format = "svg")


    def plot_error(self, error_list, num_exp, molecule = None):

        """ Plots the error over the number of experiments """

        fig, ax = plt.subplots(figsize = (5, 3))

        #ax.scatter(num_exp, error_list, c='red', label= f'Error', marker = 's',s = 20)
        ax.plot(num_exp, error_list, c='red', label= f'Error: {molecule}', marker = 's', markersize = 10, linestyle = '-')


        # graphic settings
        ax.xaxis.set_tick_params(direction='in', which='both', labelsize = 12)
        ax.yaxis.set_tick_params(direction='in', which='both', labelsize = 12)

        # set sizte of axis labels
        ax.xaxis.set_label_text("Number of Experiments", fontsize = 12)
        ax.yaxis.set_label_text("avg. Error [nm]", fontsize = 12)


        plt.tight_layout()
        #plt.show()

        # save as svg
        #fig.savefig(f"data/active_learning_error_{molecule}.svg", format = "svg")

    
    def print_parameters(self):

        """ Prints the parameters of the model """

        print(self.model)