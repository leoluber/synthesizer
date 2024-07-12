import numpy as np
import random
from Datastructure import *
from GaussianProcess import *


"""
    Run a Gaussian Process for list-format inputs from a Datastructure object
    and plots the result
"""


def main():

    ### ------------------------------------ DATA ------------------------------------ ###

    datastructure = Datastructure(
                                synthesis_file_path= "Perovskite_NC_synthesis_NH_240418.csv", 
                                target = "PLQY",                                                       # "PLQY", "FWHM", "PEAK_POS"
                                #wavelength_filter= [470, 510],                                        
                                exclude_no_star = True,
                                wavelength_unit= "NM",                                                 # "NM", "EV"
                                normalization= True,
                                )

    # standard procedure	
    data_objects = datastructure.get_data()
    parameter_selection = datastructure.total_training_parameter_selection



    # select input and target from Data objects
    inputs, targets, sample_numbers = [], [], []

    for data in data_objects:

                # INPUTS
        inputs.append(data["encoding"] + data["total_parameters"])

                # TARGETS
        targets.append(data["y"])

                # OTHER
        sample_numbers.append(data["sample_number"])



    # random sample for a MAP (3D plot), only relevant for map_3D()
    sample = random.sample(inputs, 1)[0]                                        # sample data
    sample_target = targets[inputs.index(sample)]                               # get target
    sample_number = sample_numbers[inputs.index(sample)]                        # get sample number


    # convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)



    ### ------------------------------ GAUSSIAN PROCESS ------------------------------ ###

    gp = GaussianProcess(
                        training_data = inputs,
                        parameter_selection = parameter_selection,              # for running "iterate_and_ignore()"
                        targets = targets, 
                        kernel_type = "MLP",                                    # "RBF", "MLP", "EXP", "LIN"
                        model_type  = "GPRegression",   
                        )



    ### -- choose ONE of the following options -- ###


        # (1) LOO cross validation
    #gp.choose_best_kernel()
    gp.leave_one_out_cross_validation(inputs, targets)
    gp.regression_plot()



        # (2) returns a GPy model that is trained from the specified parameters
    #gp_trained = gp.train()
    #gp.map_GP(gp_trained, "ratio", "peak pos")
    #plt.show()



        # (3) MAP (uses a random sample and varies two parameters to map the 3D space)   (ignore this)
    #model = gp.train()
    #gp.map_3D(model, sample, sample_target, sample_number, "V (antisolvent)", "c (PbBr2)")    



if __name__ == "__main__":
    main()