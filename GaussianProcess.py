import numpy as np
import matplotlib.pyplot as plt
import GPy
seed_value = 42

# custom
from Datastructure import *
from helpers import *


"""
    A class that handles all the logic for applying GPy Gaussian Processes to a list of input data points
"""


class GaussianProcess:

    def __init__(self, 
                 training_data, parameter_selection = None,
                 targets = None,
                 kernel_type = "RBF",                       # "RBF", "MLP", "EXP", "LIN"
                 model_type  = "GPRegression",              # "GPRegression"
                 ):
        

            # training specific
        self.training_data = training_data
        self.input_dim = self.training_data.shape[1]
        self.parameter_selection = parameter_selection
        self.targets = targets


            # model specific
        self.kernel_type = kernel_type
        self.kernel = None
        self.model_type = model_type
        self.model = None


            # evaluation (LOO) results
        self.loo_predictions, self.loo_uncertainty, self.loo_error = None, None, None
    


### ------------------ CROSS VALIDATION -------------------- ###


    def leave_one_out_cross_validation(self, training_data, targets):

        """
            LOO cross validation on the training data, returns the mean squared error
        """

        # LOO loop
        predictions, uncertainty, error = [], [], []

        for i in range(len(training_data)):

            print("step: "+ str(i) + "  / " + str(len(training_data)-1))


            # Split the data into training and test data and reshape so it fits the GPy standard
            X_train = np.delete(training_data, i, axis=0)
            y_train = np.delete(targets, i, axis=0)

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
            model.optimize()

            mean, variance = model.predict(X_test)

            predictions.append(mean[0][0])
            uncertainty.append(variance[0][0])
            error.append((mean[0][0] - y_test)**2)
        
        self.loo_predictions, self.loo_uncertainty, self.loo_error = np.array(predictions), np.array(uncertainty), np.array(error)

        return np.mean(self.loo_error)
    

### --------------------- OPTIMIZATION ----------------------- ###

    def train(self):
        """
            Trains the Gaussian Process model
        """

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



    def predict(self, sample):
        """
            Predicts the target value for a given sample
        """

        sample = np.reshape(sample, (1, len(sample)))
        return self.model.predict(sample)


    def iterate_and_ignore(self):
        """
            Iterates over all parameters and ignores one at a time to find the best set
        """
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
        """
            Iterates over all kernels and chooses the best one
        """

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

    def get_kernel(self, kernel, input_dim = None):
        """
            Returns the kernel object based on the kernel string
        """

        if kernel == "RBF":
            return GPy.kern.RBF(input_dim)
        elif kernel == "MLP":
            return GPy.kern.MLP(input_dim, ARD = True) + GPy.kern.MLP(input_dim, ARD=True)
        elif kernel == "EXP":
            return GPy.kern.Exponential(input_dim)
        elif kernel == "LIN":
            return GPy.kern.Linear(input_dim)
        elif kernel == "POLY":
            return GPy.kern.Poly(input_dim)
        else:
            raise ValueError("Kernel not supported")



### ------------- PLOTTING FUNCTIONS --------------- ###

    def regression_plot(self):
        """
            Plots the regression results
        """

        plt.figure(figsize=(10, 6))
        plt.scatter(self.targets, self.loo_predictions, c='r', label='Predictions')
        plt.plot([self.targets.min(), self.targets.max()], [self.targets.min(), self.targets.max()], 'k--', lw=2)

        plt.legend()
        plt.show()



    def map_GP(self, trained_model, variable, target):
        """
            Plots the GP map in parameter space
        """
        
        if self.input_dim == 1:
            x_vec = np.linspace(self.training_data.min(), self.training_data.max(), 100)

            # evaluate model for all x values
            y_vec = []
            var_vec = []
            for x in x_vec:
                y, var = trained_model.predict(np.reshape([x], (1, 1)))
                var_vec.append(var[0][0])
                y_vec.append(y[0][0])
            
            # plot
            plt.scatter(self.training_data, self.targets, c='r', label='Data')
            plt.fill_between(x_vec, np.array(y_vec) - np.array(var_vec), np.array(y_vec) + np.array(var_vec), alpha=0.5)

            plt.plot(x_vec, y_vec)
            plt.xlabel(variable)
            plt.ylabel(target)
            plt.title('GP Map')

            plt.show()

        else:
            print("Only 1D input data supported for GP map")
            return
   

    def map_3D(self, trained_model, base_sample, sample_target, sample_number, variable_1, variable_2, resolution = 30):

        """
            Plots the 3D MAP of the kernel ridge regression for two specified variables;
            extrapolates the model to a 2D grid with target as the z-axis
        """

        if variable_1 not in self.parameter_selection or variable_2 not in self.parameter_selection:
            print("Variable not in parameter selection")
            return
        
        if len(self.training_data[0]) < 2:
            print("2D MAP only supported for >=2D data")
            return


        # get the indices of the variables
        index_1 = self.parameter_selection.index(variable_1)
        index_2 = self.parameter_selection.index(variable_2)

        data_1 = self.training_data[:, index_1]
        data_2 = self.training_data[:, index_2]
        data_y = self.targets


        # original coordinates
        x, y = base_sample[index_1], base_sample[index_2]


        # create the grid
        x_vec = np.linspace(0,1, resolution)
        y_vec = np.linspace(0,1, resolution)
        X, Y = np.meshgrid(x_vec, y_vec)


        # iterate over the grid and predict the values
        Z, Z_max, Z_min = np.zeros((resolution, resolution)), np.zeros((resolution, resolution)), np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                base_sample[index_1] = x_vec[i]
                base_sample[index_2] = y_vec[j]

                Z[i, j], var = trained_model.predict(np.reshape([base_sample], (1, len(base_sample))))
                Z_max[i, j] = Z[i, j] + var
                Z_min[i, j] = Z[i, j] - var


        # plot the results in 3D
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, sample_target, c='r', label='Data')
        ax.scatter3D(data_1, data_2, data_y, c='b', label='Data')

        ax.plot_surface(X, Y, Z, cmap='RdGy', alpha=0.8)
        ax.plot_surface(X, Y, Z_max, color='gray', alpha=0.5)
        ax.plot_surface(X, Y, Z_min, color='gray', alpha=0.5)

        ax.set_xlabel(variable_1)
        ax.set_ylabel(variable_2)
        ax.set_zlabel('Target')
        plt.title(f"3D MAP for sample {sample_number}")
        plt.show()
