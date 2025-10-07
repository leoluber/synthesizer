
""" 
    File:        run_Synthesizer.py
    Project:     Synthesizer: Chemistry-Aware Machine Learning for 
                 Precision Control of Nanocrystal Growth 
                 (Henke et al., Advanced Materials 2025)
    Description: This script uses the Synthesizer.py module to optimize Perovskite NCs
                 for a given antisolvent molecule and objective function
    Author:      << github.com/leoluber >> 
    License:     MIT
"""

# -------
import os
import sys
# -------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from package.src.Synthesizer import Synthesizer



""" Uses the Synthesizer class to optimize Perovskite NCs for a given 
    antisolvent molecule

SUMMARY
-------
- GP regression models for Peak Position, PLQY, FWHM, ...
- bounds given by min/max of all data points or custom bounds
- uses GPyOpt for optimization

"""


### -------------------- choose molecule and target peak -------------------- ###
antisolvent =  "Methanol"
target_peak   =   500  #nm
### ------------------------------------------------------------------------- ###


def main():
    synthesizer = Synthesizer(antisolvent, 
                              data_path=         "dataset_synthesizer.csv",
                              iterations =       100, 
                              peak =             target_peak,
                              obj =              ["peak_pos", "fwhm"], 

                              # fixed parameters (optional, otherwise set to None)
                              c_Cs_fixed =       0.02,
                              V_Cs_fixed =       100,

                              # experimental boundaries (optional, otherwise set to None)
                              V_As_max =         5000,
                              V_Pb_max =         2000,

                              add_baseline=      True,
                              )
    

    # optimize NC
    opt_x, opt_delta = synthesizer.optimize_NC()

    # get results
    results_string =    synthesizer.results["results_string"]
    As_Pb_ratio=        synthesizer.results["As_Pb_ratio"]
    Cs_Pb_ratio =       synthesizer.results["Cs_Pb_ratio"]

    # handle results 
    synthesizer.print_results( results_string=results_string,
                               peak_pos=synthesizer.results["pred_peak"],
                               As_Pb_ratio=As_Pb_ratio,
                               Cs_Pb_ratio=Cs_Pb_ratio,
                            )


if __name__ == "__main__":
    main()
