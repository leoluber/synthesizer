r""" Script to optimize NPLs for a given antisolvent molecule (target: PLQY) 
    using the Synthesizer.py module. """
    # < github.com/leoluber >


# custom
from Synthesizer import Synthesizer
from helpers import *



"""
    This script uses the Synthesizer.py module to optimize Perovskite NPLs for a given 
    antisolvent molecule (target: PLQY)
    - KRR / GP regression models for Peak Position, PLQY, FWHM, ...
    - bounds given by min/max of all data points
    - uses GPyOpt for optimization
"""



### -------------------- choose molecule and target peak -------------------- ###

#molecule_name = input("Enter the molecule name: (e.g. Ethanol, Methanol, ...):     ")
#target_peak = int(input("Enter the target peak position in nm: (e.g. 470):         "))

molecule_name =  "Methanol"
geometry =        [0.5, 1, 0]
target_peak   =   496

### ------------------------------------------------------------------------- ###



def main():

    """ For explanation of the parameters see the Synthesizer.py module """

    # initialize synthesizer object and optimize (specify As molecule and NPL type)
    synthesizer = Synthesizer(molecule_name, 
                              iterations =       10, 
                              peak =             target_peak,
                              obj =              ["PEAK_POS", "PLQY",], 
                              encoding_type =    "one_hot", 
                              #geometry =        geometry,
                              Cs_Pb_opt =        False,
                              Cs_As_opt=         False,
                              c_Pb_fixed =       0.05, 
                              #V_As_fixed=       5000,
                              #V_Cs_fixed=       100,  
                              c_Pb_max =         None,
                              model_type=       "GP",
                              )
    
    synthesizer.print_logo()
    
    # optimize NPL
    opt_x, opt_delta = synthesizer.optimize_NPL()
    print(f"\n")
    print(f"opt_delta: {opt_delta}")


    # get results
    results_string =    synthesizer.results["results_string"]
    input_NPL =         synthesizer.results["input_NPL"]
    input_PLQY =        synthesizer.results["input_PLQY"]
    input_FWHM =        synthesizer.results["input_FWHM"]
    As_Pb_ratio=        synthesizer.results["As_Pb_ratio"]
    Cs_Pb_ratio =       synthesizer.results["Cs_Pb_ratio"]


    # handle results (all done by the synthesizer class)
    synthesizer.print_results(results_string, 
                              input_PLQY, 
                              input_NPL, 
                              input_FWHM, 
                              As_Pb_ratio, 
                              Cs_Pb_ratio)
    #synthesizer.test_results(input_PLQY)       



if __name__ == "__main__":
    main()

