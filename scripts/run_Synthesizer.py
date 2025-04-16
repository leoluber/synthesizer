""" 
    Project:     synthesizer
    File:        run_Synthesizer.py
    Description: This script uses the Synthesizer.py module to optimize Perovskite NPLs 
                 for a given antisolvent molecule and objective function
    Author:      << github.com/leoluber >> 
    License:     MIT
"""



import os
import sys


# custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from package.src.Synthesizer import Synthesizer






""" Uses the Synthesizer class to optimize Perovskite NPLs for a given 
    antisolvent molecule

SUMMARY
-------

- GP regression models for Peak Position, PLQY, FWHM, ...
- bounds given by min/max of all data points or custom bounds
- uses GPyOpt for optimization

"""






### -------------------- choose molecule and target peak -------------------- ###

antisolvent =  "Acetone"
target_peak   =   461

### ------------------------------------------------------------------------- ###



def main():

    synthesizer = Synthesizer(antisolvent, 
                              
                            #   data_path=        "CsPbI3_NH_LB_AS_BS_combined_new.csv",
                            #   spectral_path=    "spectrum_CsPbI3/",
                              data_path=         "Perovskite_NC_synthesis_NH_240418_transfer.csv",
                              spectral_path=     "spectrum/",


                              iterations =       100, 
                              peak =             target_peak,
                              obj =              ["peak_pos", "fwhm"], 
                              ion=               "CsPbBr3",  # "CsPbI3", "CsPbBr3",
                              Cs_Pb_opt =        False,
                              Cs_As_opt=         False,
                              c_Pb_fixed =       0.01, 
                              c_Cs_fixed =       0.02,
                              V_Cs_fixed=        100,  
                              c_Pb_max =         None,
                              V_As_max=          5000,
                              V_Pb_max=          2000,
                              add_baseline=      True,
                              )
    
    
    #synthesizer.print_logo()
    
    # optimize NPL
    opt_x, opt_delta = synthesizer.optimize_NPL()
    print(f"\n")
    print(f"opt_delta: {opt_delta}")


    # get results
    results_string =    synthesizer.results["results_string"]
    As_Pb_ratio=        synthesizer.results["As_Pb_ratio"]
    Cs_Pb_ratio =       synthesizer.results["Cs_Pb_ratio"]


    # handle results (all done by the synthesizer class)
    synthesizer.print_results( results_string=results_string,
                               peak_pos=synthesizer.results["pred_peak"],
                               As_Pb_ratio=As_Pb_ratio,
                               Cs_Pb_ratio=Cs_Pb_ratio,
                            )





if __name__ == "__main__":
    main()

