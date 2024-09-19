""" Kernel Ridge Regression on a Datastructure object using the KRR from sci-kit learn """
    # < github.com/leoluber >


from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from typing import Literal

#custom
from helpers import *





class Ridge:

    """ Kernel Ridge Regression:

    BASICS
    ------
    - uses scikit-learn's KernelRidge
    - can optimize hyperparameters
    - can optimize the choice of input parameters
    - can plot regression results
    - can plot 3D MAPs in parameter space

    PARAMETERS
    ----------
    - training_data: list of training data (list)
    - targets: list of targets (list)
    - parameter_selection: list of parameter names (list)
    - kernel_type: type of kernel (str)
    - alpha: regularization parameter (float)
    - gamma: kernel coefficient (float)

    USAGE
    -----
    >>> rk = Ridge(training_data, targets, parameter_selection, ...)
    >>> rk.optimize_hyperparameters()
    >>> rk.fit()
    >>> rk.loo_validation()
    >>> rk.predict(...)

    """


    def __init__(
                 self, 
                 training_data: list,
                 targets: list,
                 parameter_selection: list,
                 kernel_type: Literal["laplacian", "rbf", "linear", "polynomial"] = "laplacian",
                 alpha      = 0.01,
                 gamma      = 0.01,
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

            # performance
        self.fitting_time =         None




### -------------------------- STANDARD -------------------------- ###

    def fit(self):
        return self.model.fit(self.training_data, self.targets)

    def predict(self, X):
        return self.model.predict(X)


### ----------------------- LOO Validation ----------------------- ###

    def loo_validation(self, inputs, targets, molecule = None) -> float:
        """LOO cross validation on the training data"""

        self.predictions, self.error = [], []

        for i in range(len(inputs)):

            # leave one out
            train_inputs  = inputs[:i] + inputs[i+1:]
            train_targets = targets[:i] + targets[i+1:]

            # timed fitting for the first iteration
            if self.fitting_time is None: 
                timer_start = timer()

            self.model.fit(train_inputs, train_targets)

            if self.fitting_time is None:
                self.fitting_time = timer() - timer_start

            prediction = self.model.predict([inputs[i]])

            # store predictions and errors
            self.predictions.append(prediction[0])
            self.error.append(abs(prediction - targets[i]))
        

        # print results
        print(f"mean error: {np.mean(self.error)}")
        print(f"fitting time: {self.fitting_time}")
        
        #plot regression
        self.plot_regression(targets, self.predictions, molecule)
        return np.mean(self.error)



    def transfer_validation(self, transfer_inputs, transfer_targets):
        
        """ Validation on a separate dataset """

        predictions, errors = [], []

        for i, input in enumerate(transfer_inputs):
            prediction = self.model.predict([input])
            predictions.append(prediction[0])

            errors.append(abs(prediction - transfer_targets[i]))

        self.plot_regression(transfer_targets, predictions)

        return 0



### ------------------------ OPTIMIZATION ------------------------ ###


    def optimize_hyperparameters(self) -> KernelRidge:
        
        """ Optimize the hyperparameters of the kernel ridge regression """


        krr = KernelRidge()
        param_grid = {
            'alpha':  [0.1, 0.01, 0.001],
            'gamma':  [0.01,],
            'kernel': ['laplacian',]
        }

        # grid search for hyperparameters
        gs_krr = GridSearchCV(estimator=krr, 
                              param_grid=param_grid, 
                              scoring='neg_root_mean_squared_error', 
                              cv= int(self.num_data))


        gs_krr.fit(self.training_data[:int(self.num_data)], self.targets[:int(self.num_data)])

        print(gs_krr.best_params_)

        # get the best model hyperparameters
        best_krr = gs_krr.best_estimator_

        self.model = best_krr
        return best_krr



### ------------------------ PLOTTING ------------------------ ###

    def plot_regression(self, targets = None, predictions = None, molecule = None):
        
        """ Plots the regression results """

        # default is the loo output
        if targets == None: targets = self.targets
        if predictions == None: predictions = self.predictions


        plt.scatter(targets, predictions, color='blue', label='krr', s=10)
        plt.plot([min(targets), max(targets)], 
                 [min(targets), max(targets)], 
                 color='black', linestyle='--')

        if molecule != None: plt.title(f'KRR - LOO - {molecule}')
        plt.legend()
        plt.show()
    


    def plot_1D(self, initial_sample, changed_index,
                range = [0,10], resolution = 100):
        
        """ Plots the 1D MAP of the kernel ridge regression for the specified variable """

        lin = np.linspace(range[0], range[1], resolution)

        inputs  = [initial_sample[:changed_index] + [x] + initial_sample[changed_index+1:] for x in lin]
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