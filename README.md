
# Synthesizer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)


## Overview

**The Synthesizer** is a package that optimizes the experimental parameters for 
antisolvent controlled perovskite NC synthesis. It trains Gaussian Process 
Regressors for a selection of targets and optimizes volumes and concentrations 
as well as other arbitrary experimental parameters.

It is published as "Synthesizer: Chemistry-Aware Machine Learning for Precision
Control of Nanocrystal Growth". (Advanced Materials, 2025)


### Features
- 📌 Feature 1: Feature selection and calculation
- 📌 Feature 2: 1-3nm accurate PL peak position prediction from those features
- 📌 Feature 3: Simultaneous optimization for arbitrary set of targets
- 📌 Feature 4: High transfer capabilities to unknown antisolvents

- 📌 Can be used with the provided dataset or applied to your own!


## Usage

Some setup steps are required to adapt the **Synthesizer** to your project.

### Setup Datastructure

A table of synthesis parameters (in the format of "data\raw\Perovskite_NC_synthesis_NH_240418_new.csv") needs to be provided. The dictionaries/tables at "data\raw\AntisolventProperties.csv", "data\raw\molecule_encoding.json" and "data\raw\molecule_dictionary.json" might need to be updated.

Additionally, any potential custom properties need to be added both in Datastructure.__init__ and Datastructure.calculate_properties.

### The Geometric Encoding

For details see the publication referenced above. The translation of antisolvent molecule to encoding can be found in data\raw\molecule_encoding.json and should also be adjusted there for other applications.


### Train a Model

The Gaussian Process pipline as well as the plotting functionality can be tested via the run_GP.py script. 


### Synthesizer Optimization

Set targets and general requirements in the run_Synthesizer.py file and run the file!

```bash
run_Synthesizer.py
```


## Project Structure

```
synthesizer/
│── package/                    # Source code
│   │── __init__.py             # Package initialization
│   │── src/
|      |── Datastructure.py 
|      |── GaussianProcess.py
|      |── Synthesizer.py
│
│── plotting/                   
|   |── Plotter.py              # handles all visualization
|   
│── scripts/                    # Example scripts
|   |── run_Synthesizer.py      # optimize NCs
|   |── run_GP.py               # testing setup and plotting models
│
│── output/   
|   |── suggestions.txt         # contains the resulting synthesis suggestions                
│
│── README.md                   # Documentation
│── setup.py                    # setup file
│── LICENSE.txt                 # License file
```




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, reach out to:
📧 **Leo Luber** - l.luber@campus.lmu.de
🌐 [GitHub Profile](https://github.com/leoluber)

