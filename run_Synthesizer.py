import numpy as np

# custom
from Datastructure import Datastructure
from Synthesizer import Synthesizer
from helpers import *


"""
    Uses the "Synthesizer" module to optimize Perovskite NPLs for a given antisolvent molecule (target: PLQY)
    - KRR regression models for #MLs and PLQY
    - bounds given by min/max of all data points
    - uses GPyOpt for optimization
    - constraints taken from the #ML model
    - objective function: (1-PLQY)**2 + (target_peak_pos - peak_pos)**2 + FWHM        
                      OR: (1-PLQY)**2 * (target_peak_pos - peak_pos)**2 * FWHM 
"""


### -------------------- choose molecule and target peak -------------------- ###

#molecule_name = input("Enter the molecule name: (e.g. Ethanol, Methanol, ...):     ")
#target_peak = int(input("Enter the target peak position in nm: (e.g. 470):         "))

molecule_name = "Ethanol"
target_peak = 480

### ------------------------------------------------------------------------- ###



def main():

    # initialize synthesizer object and optimize (specify As molecule and NPL type)
    synthesizer = Synthesizer(molecule_name, iterations=100, peak = target_peak, obj=["PEAK_POS", "PLQY"], encoding_type="one_hot")
    opt_x, opt_delta = synthesizer.optimize_NPL()

    print(f"opt_x: {opt_x}")
    print(f"opt_delta: {opt_delta}")

    # get results
    results_string =        synthesizer.results["results_string"]
    input_NPL =             synthesizer.results["input_NPL"]
    input_PLQY =            synthesizer.results["input_PLQY"]
    input_FWHM =            synthesizer.results["input_FWHM"]
    As_Pb_ratio=            synthesizer.results["As_Pb_ratio"]


    # handle results (all done by the synthesizer class)
    synthesizer.print_results(results_string, input_PLQY, input_NPL, input_FWHM, As_Pb_ratio)       	  # -->  print results
    synthesizer.test_results(input_PLQY)                                                                  # -->  test results against Gaussian Process
    synthesizer.plot_suggestions(opt_x, synthesizer.datastructure_PLQY, parameters=synthesizer.datastructure_PLQY.synthesis_training_selection)
    plt.show()


if __name__ == "__main__":
    main()

