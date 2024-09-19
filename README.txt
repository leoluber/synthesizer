
# SYNTHESIZER

The SYNTHESIZER is an Optimization module that samples synthesis parameters for 
Perovskite NPLs to maximize PLQY, FWHM (and other targets)



## BASIC USAGE

Run "run_Synthesizer.py" and specify the requested choices of antisolvent 
molecule and target peak position
 - the synthesis training data (data\Perovskite_NC_synthesis_NH_240418.csv) 
   as well as the spectrum folder (data\spectrum) need to be at that path
 - the results will be written to "suggestions.txt" in the same folder



## CLASSES AND THEIR PURPOSE

- Synthesizer:      Optimization class (based on GPy Opt)
- Datastructure:    handles all data related functionality (reading, selection, 
  normalization, parameter computations and so on)
- KRR:              implements the Kernel Ridge Regression model from sklearn 
  with extra logic
- GaussianProcess:  implements the GPy Gaussian Process with some extra logic



## REQUIREMENTS

- torch, torch_geometric
- GPy, GPyOpt
- sklearn
- plotly