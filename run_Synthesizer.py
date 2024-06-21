import numpy as np
from Datastructure import Datastructure
from Synthesizer import Synthesizer
from helpers import *

"""
    Uses the "Synthesizer" module to optimize Perovskite NPLs for a given molecule (target: PLQY)
    - KRR regression models for #MLs and PLQY
    - bounds given by min/max of all data points
    - uses GPyOpt for optimization
    - constraints taken from the #ML model
    - objective function: (1-PLQY)**2 * (target_peak_pos - peak_pos)**2
"""


### -------------------- choose molecule and target peak -------------------- ###
molecule_name = input("Enter the molecule name: (e.g. Ethanol, Methanol, ...):     ")
target_peak = int(input("Enter the target peak position in nm: (e.g. 470):     "))
### ------------------------------------------------------------------------- ###

def main():

    # initialize synthesizer object and optimize (specify As molecule and NPL type)
    synthesizer = Synthesizer(molecule_name, iterations=50, peak = target_peak)
    opt_x, opt_delta = synthesizer.optimize_NPL()
    print(f"opt_x: {opt_x}")
    print(f"opt_delta: {opt_delta}")

    # get As/Pb ratio
    max_As_Pb_ratio =  synthesizer.datastructure_NPL.max_min["AS_Pb_ratio"][0]
    As_Pb_ratio =      opt_x[0] / (opt_x[1] * opt_x[2] * max_As_Pb_ratio)


    # get initial input that leads to the best plqy value
    input =             np.array(synthesizer.one_hot_molecule)
    peak_pos_input =    np.append(input, As_Pb_ratio)
    input =             np.append(input, opt_x)
    x =                 [value for value in input]
    x_peak_pos =        [value for value in peak_pos_input]


    # denormalize the parameters
    results_string = []
    denorm = {}
    for i, parameter in enumerate(synthesizer.parameters_PLQY):
        denorm[parameter] = synthesizer.datastructure_PLQY.denormalize(opt_x[i], parameter)
        results_string.append(f"{parameter} :  {denorm[parameter]}")
    As_Pb_ratio_denorm = denorm["V (antisolvent)"] / (denorm["c (PbBr2)"] * denorm["V (PbBr2 prec.)"])


    # handle results (all done by the synthesizer class)
    synthesizer.print_results(results_string, x, x_peak_pos, As_Pb_ratio_denorm)                 #-->  print results
    synthesizer.test_results(x)                                                                  #-->  test results against Gaussian Process
    synthesizer.plot_suggestions(opt_x[:3], synthesizer.datastructure_PLQY, parameters=synthesizer.datastructure_PLQY.synthesis_training_selection[:3])
    plt.show()



if __name__ == "__main__":
    main()

