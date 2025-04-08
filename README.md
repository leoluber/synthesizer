
# Synthesizer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-brightgreen)


## Overview

**The Synthesizer** is a package that optimizes the experimental parameters for 
antisolvent controlled perovskite NPL synthesis. It trains Gaussian Process 
Regressors for a selection of targets and optimizes volumes and concentrations 
as well as other arbitrary experimental parameters.


### Features
- 📌 Feature 1: Feature selection and calculation
- 📌 Feature 2: 1-3nm accurate PL peak position prediction from those features
- 📌 Feature 3: Simultaneous optimization for arbitrary set of targets
- 📌 Feature 4: High transfer capabilities to unknown antisolvents

- 📌 Can be used with the provided dataset or applied to your own!



## Installation

### 1️⃣ Clone the Repository
Since this package is not available via `pip`, you need to clone it manually:

```bash
git clone https://github.com/leoluber/synthesizer
cd my_package
```

### 2️⃣ Install Dependencies

Ensure you have Python **3.12+** installed, then install the required dependencies:

```bash
python setup.py install
```


## Usage

After installation, you can start using the package in your Python scripts.

### Setup Datastructure

(...)

### Synthesizer Optimization

Run the file

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
|      |── Preprocessor.py
|      |── GaussianProcess.py
|      |── helpers.py
|      |── Synthesizer.py
|   
│
│── scripts/                    # Example scripts
|   |── run_Synthesizer.py
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

