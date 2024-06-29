from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np
from helpers import *
import matplotlib.pyplot as plt


"""
    Kernel Ridge Regression:
    - uses scikit-learn's KernelRidge
    - can optimize hyperparameters
    - can optimize the choice of input parameters
    - can plot regression results
    - can plot 3D MAPs in parameter space
"""


class Ridge:
    def __init__(self, 
                 training_data,
                 targets,
                 parameter_selection,       # list of parameter names
                 kernel_type = "rbf",       # "rbf", "polynomial", "laplacian", "linear"
                 alpha = 0.1,               # regularization parameter
                 gamma = 0.1,               # kernel coefficient
                 ):
        

            # training specific
        self.training_data = training_data
        self.input_dim = len(self.training_data[0])
        self.num_data = len(targets)
        self.targets = targets
        self.parameter_selection = parameter_selection

            # model specific
        
        self.model = KernelRidge(kernel=kernel_type, alpha=alpha, gamma=gamma)
        self.predictions, self.error = [],[]


### -------------------------- STANDARD -------------------------- ###

    def fit(self):
        self.model.fit(self.training_data, self.targets)
        return self.model


    def predict(self, X):
        return self.model.predict(X)

### ----------------------- LOO Validation ----------------------- ###

    def loo_validation(self, inputs, targets):
        """
            LOO cross validation on the training data
        """
        self.predictions, self.error = [], []
        for i in range(len(inputs)):

            #print(f"LOO iteration: {i}")
            train_inputs = inputs[:i] + inputs[i+1:]
            train_targets = targets[:i] + targets[i+1:]

            self.model.fit(train_inputs, train_targets)

            prediction = self.model.predict([inputs[i]])
            self.predictions.append(prediction[0])
            self.error.append(abs(prediction - targets[i]))
        return self.predictions, np.mean(self.error)


### ------------------------ OPTIMIZATION ------------------------ ###

    def optimize_hyperparameters(self):
        """
            Optimize the hyperparameters of the kernel ridge regression
        """
        krr = KernelRidge()
        param_grid = {
            'alpha': [1e-1, 1e-3, 1e-5, 1e-7, 1e-8],
            'gamma': [0.5, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'polynomial', 'laplacian', 'linear']
        }

            # grid search for hyperparameters
        gs_krr = GridSearchCV(estimator=krr, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv= int(self.num_data))
        gs_krr.fit(self.training_data[:int(self.num_data)], self.targets[:int(self.num_data)])

        print(gs_krr.best_params_)

        best_krr = gs_krr.best_estimator_
        self.model = best_krr
        return best_krr


    def optimize_inputs(self, iterations = 1):
        """
            Optimize the choice of parameters by excluding them one by one
            and checking if the error decreases
        """
            
        # main opt. loop
        for i in range(iterations):
            print(f"parameter selection: {self.parameter_selection}")
                
            # first we run the loo validation on the full dataset to get a baseline
            base_predictions, smallest_error = self.loo_validation(self.training_data, self.targets)
            best_parameter_selection = self.parameter_selection
            training_data = self.training_data

            # then we exclude one parameter at a time and check if the error decreases
            for index, parameter in enumerate(self.parameter_selection):

                # we need a local copy of the data
                local_inputs, local_targets, local_parameter_selection = self.training_data, self.targets, self.parameter_selection
                local_inputs = [data[:index] + data[index+1:] for data in local_inputs]
                local_parameter_selection = local_parameter_selection[:index] + local_parameter_selection[index+1:]
                    
                # run the loo validation
                predictions, mean_error = self.loo_validation(local_inputs, local_targets)

                # if the error is smaller, we update the best parameter selection
                if mean_error < smallest_error:
                    smallest_error = mean_error
                    best_parameter_selection = local_parameter_selection
                    training_data = local_inputs
            
            self.parameter_selection = best_parameter_selection
            self.training_data = training_data
        

        print(f"best parameter selection: {best_parameter_selection}")


            


### ------------------------ PLOTTING ------------------------ ###

    def plot_regression(self):
        """
            Plots the regression results
        """
        print(f"targets: {len(self.targets)}")
        print(f"predictions: {len(self.predictions)}")
        plt.scatter(self.targets, self.predictions, color='blue', label='krr')
        plt.plot([min(self.targets), max(self.targets)], [min(self.targets), max(self.targets)], color='black', linestyle='--')
        plt.title('KRR - LOO')
        plt.legend()
        plt.show()
    
    
    def map_3D(self, base_sample, sample_target, sample_number, variable_1, variable_2, resolution = 30):
        """
            Plots the 2D MAP of the kernel ridge regression for two specified variables
        """

        if variable_1 not in self.parameter_selection or variable_2 not in self.parameter_selection:
            print("Variable not in parameter selection")
            return
        
        if len(self.training_data[0]) < 2:
            print("2D MAP only supported for 2D data")
            return


        # get the indices of the variables
        index_1 = self.parameter_selection.index(variable_1)
        index_2 = self.parameter_selection.index(variable_2)


        # original coordinates
        x, y = base_sample[index_1], base_sample[index_2]
        x_vec = np.linspace(0,1, resolution)
        y_vec = np.linspace(0,1, resolution)
        X, Y = np.meshgrid(x_vec, y_vec)


        # iterate over the grid and predict the values
        z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                base_sample[index_1] = x_vec[i]
                base_sample[index_2] = y_vec[j]
                z[i,j] = self.model.predict([base_sample])


        # plot the results in 3D
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, sample_target, c='r', label='Data')
        ax.plot_surface(X, Y, z, cmap='RdGy', alpha=0.8)
        ax.set_xlabel(variable_1)
        ax.set_ylabel(variable_2)
        ax.set_zlabel('Target')
        plt.title(f"3D MAP for sample {sample_number}")
        plt.show()

    def plot_1D(self, resolution = 30):
        """
            Plots the 1D MAP of the kernel ridge regression for the specified variable
        """

        if len(self.training_data[0]) > 1:
            print("1D MAP only supported for 1D data")
            return

        x_vec = np.linspace(min(self.training_data), max(self.training_data), resolution)
        y_vec = [self.model.predict([x]) for x in x_vec]

        plt.scatter(self.training_data, self.targets, color='blue', label='Data')
        plt.plot(x_vec, y_vec, label='KRR')
        plt.title('KRR - 1D MAP')
        #plt.legend()
        #plt.show()


    def visualize_kernel(self, ax, threshhold = [0.2, 0.3, 0.4]):
        """
            Visualizes the kernel function
            (only for 3D data)
        """
        X = np.linspace(0, 0.5, 30)
        Y = np.linspace(0, 0.5, 30)
        Z = np.linspace(0, 1, 30)
        sheet_04 = np.ones((30, 30))
        sheet_03 = np.ones((30, 30))
        sheet_02 = np.ones((30, 30))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for z in Z:
                    plqy = self.model.predict([[x, y, z]])
                    for thresh in threshhold:
                        if np.round(plqy, 1) == thresh:
                            if thresh == 0.2:
                                sheet_02[i,j] = z
                            elif thresh == 0.3:
                                sheet_03[i,j] = z
                            elif thresh == 0.4:
                                sheet_04[i,j] = z
                            break
                for sheet in [sheet_02, sheet_03, sheet_04]:
                    if sheet[i,j] == 1: sheet[i,j] = np.nan
        
        X, Y = np.meshgrid(X, Y)
        for sheet, threshhold in zip([sheet_02, sheet_03, sheet_04], threshhold):
            ax.plot_surface(X, Y, sheet, alpha=0.4, label = f"Pred. PLQY = {threshhold}")
        plt.legend()

        plt.show()