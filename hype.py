""" hyperparameter optimization for Gaussian Process Molecular Transfer """


import GPy
import GPyOpt
from Datastructure import Datastructure
import numpy as np



r"""

HYPERPARAMETER OPTIMIZATION FOR GAUSSIAN PROCESS MOLECULAR TRANSFER
___________________________________________________________________

This script is used to optimize the hyperparameters of the Gaussian Process
for molecular transfer. The main focus is on the kernel composition and the 
geometric encoding of the molecules.

The optimization is done using the GPyOpt package for a single molecule 
transfer.

"""



# Settings
TRANSFER_MOLECULE = "Methanol"



# get the data 
datastructure = Datastructure(
    synthesis_file_path = "Transfer_Data.csv", 
    target              = "PEAK_POS",                                     
    wavelength_unit     = "NM",
    monodispersity_only = True,
    encoding            = "geometry", 
    P_only              = True,
    molecule            = "all",
    add_baseline        = True,
    )

datastructure.synthesis_training_selection  = ["AS_Pb_ratio", "Cs_Pb_ratio",]
data_objects = datastructure.get_data()



# split the data into training and test data
global x_train, y_train, x_test, y_test
x_train = np.array([data["encoding"] + data["total_parameters"] for data in data_objects if data["molecule_name"] != TRANSFER_MOLECULE])
y_train = np.array([data["y"] for data in data_objects if data["molecule_name"] != TRANSFER_MOLECULE])

x_test = np.array([data["encoding"] + data["total_parameters"] for data in data_objects if data["molecule_name"] == TRANSFER_MOLECULE])
y_test = np.array([data["y"] for data in data_objects if data ["molecule_name"] == TRANSFER_MOLECULE])






# define the optimization function

BOUNDS = [ {"name": "lower_length_1", "type": "continuous", "domain": (0, 400)},
           {"name": "upper_length_1", "type": "continuous", "domain": (401, 1000)},
           {"name": "lower_length_2", "type": "continuous", "domain": (0, 10)},
           {"name": "upper_length_2", "type": "continuous", "domain": (11, 100)},]



def optimize_GP(params):

    """
    Optimizes the Gaussian Process hyperparameters for molecular transfer
    """

    print(f"Optimizing with parameters: {params}")

    global x_train, y_train, x_test, y_test

    # reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    # construct the two kernels for the parameter space and the geometry space
    parameter_kernel = GPy.kern.Exponential(2, active_dims= [5,6])
    parameter_kernel.lengthscale.constrain_bounded(params[0][0], params[0][1])
    parameter_kernel.variance.constrain_bounded(100, 600)

    geometry_kernel =  GPy.kern.Exponential(5, active_dims=[0,1,2,3,4,],)
    geometry_kernel.lengthscale.constrain_bounded(params[0][2], params[0][3])
    geometry_kernel.variance.constrain_bounded(0.1, 200)


    # combine the kernels
    kernel = parameter_kernel + geometry_kernel

    # create the model
    model = GPy.models.GPRegression(x_train, y_train, kernel)
    model.optimize()


    # predict the test data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))	
    y_pred, y_var = model.predict(x_test)

    # calculate the error
    error = np.mean(np.abs(y_pred - y_test))

    return error



# optimize the hyperparameters
optimizer = GPyOpt.methods.BayesianOptimization(f = optimize_GP, domain = BOUNDS)
optimizer.run_optimization(max_iter = 5)

print(optimizer.x_opt)


# train the model with the optimized hyperparameters
# TODO


    





                            