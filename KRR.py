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
                 kernel_type = None,       # "rbf", "polynomial", "laplacian", "linear"
                 alpha = 0.1,               # regularization parameter
                 gamma = 0.1,               # kernel coefficient
                 ):
        

            # training specific
        self.training_data =        training_data
        self.input_dim =            len(self.training_data[0])
        self.num_data =             len(targets)
        self.targets =              targets
        self.parameter_selection =  parameter_selection

            
            # model specific
        self.model =                KernelRidge(kernel=kernel_type, alpha=alpha, gamma=gamma)
        self.predictions =          []
        self.error =                []




### -------------------------- STANDARD -------------------------- ###

    def fit(self):
        return self.model.fit(self.training_data, self.targets)

    def predict(self, X):
        return self.model.predict(X)



### ----------------------- LOO Validation ----------------------- ###

    def loo_validation(self, inputs, targets) -> float:
        """
            LOO cross validation on the training data
        """
        self.predictions, self.error = [], []

        for i in range(len(inputs)):

            train_inputs = inputs[:i] + inputs[i+1:]
            train_targets = targets[:i] + targets[i+1:]

            self.model.fit(train_inputs, train_targets)

            prediction = self.model.predict([inputs[i]])

            self.predictions.append(prediction[0])
            self.error.append(abs(prediction - targets[i]))
        

        #plot regression
        self.plot_regression(targets, self.predictions)

        return np.mean(self.error)



    def transfer_validation(self, transfer_inputs, transfer_targets):
        """
            Validation on a separate dataset
        """

        predictions, errors = [], []

        for i, input in enumerate(transfer_inputs):
            prediction = self.model.predict([input])
            predictions.append(prediction[0])

            errors.append(abs(prediction - transfer_targets[i]))

        self.plot_regression(transfer_targets, predictions)

        return 0



### ------------------------ OPTIMIZATION ------------------------ ###

    def optimize_hyperparameters(self) -> KernelRidge:
        """
            Optimize the hyperparameters of the kernel ridge regression
        """

        krr = KernelRidge()

        param_grid = {
            'alpha': [ 1e-3, 1e-2, 1e-1,],
            'gamma': [ 0.1, 0.0001, 0.001,],
            'kernel': ['laplacian', 'exponential',]
        }

        # grid search for hyperparameters
        gs_krr = GridSearchCV(estimator=krr, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv= int(self.num_data))
        gs_krr.fit(self.training_data[:int(self.num_data)], self.targets[:int(self.num_data)])

        print(gs_krr.best_params_)

        # get the best model hyperparameters
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

    def plot_regression(self, targets = None, predictions = None):
        """
            Plots the regression results
        """

        # default is the loo output
        if targets == None: targets = self.targets
        if predictions == None: predictions = self.predictions

        print(f"targets: {len(targets)}")
        print(f"predictions: {len(predictions)}")

        plt.scatter(targets, predictions, color='blue', label='krr')
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color='black', linestyle='--')

        plt.title('KRR - LOO')
        plt.legend()
        plt.show()
    


    def plot_1D(self, initial_sample, changed_index, range = [0,10], resolution = 100):
        """
            Plots the 1D MAP of the kernel ridge regression for the specified variable
        """

        lin = np.linspace(range[0], range[1], resolution)

        inputs = [initial_sample[:changed_index] + [x] + initial_sample[changed_index+1:] for x in lin]
        targets = [self.model.predict([x]) for x in inputs]

        plt.plot(lin, targets, color='red', label='KRR')
        plt.scatter(self.training_data, self.targets, color='blue', label='data')
        plt.title('KRR - 1D MAP')
        
        #plt.show()



    def visualize_kernel(self, ax, threshhold = [0.2, 0.3, 0.4]):
        """
            Visualizes the kernel function as shells of constant PLQY
            (only for 3D data)
            ALSO: the code is a mess, i know ...
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