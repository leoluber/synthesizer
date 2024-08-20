import numpy as np
from plotly import graph_objects as go
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from scipy.optimize import curve_fit
import GPy

# custom
import Datastructure


"""
    This model class is predicts perovskite NPL properties for unknown molecules
    by using a mixture of experts approach combined with a kernel ridge regression model.
    A transfer is achieved by training the model on a set of various molecules and using
    a metric of similarity to predict the properties of a new molecule.
    - The model is trained on a set of synthesis data of perovskite NPLs
    - The model is validated on a set of unseen data
    - functionality for physical interpretation of the model to be added...
    ALSO: keep in mind only a fraction of all defined functions are used in each setting

    TODO: generalize the sigmoid fit bounds, scipy curve_fit is crime against humanity
"""



class GeometricTransfer:


    def __init__(self,
                 molecule_training_selection: list,                                     # list of molecules to train on
                 expert =           "KRR",                                              # "KRR", "SIGMOID_FILTERS", "GP"
                 predictor =        "MLP",                                              # "KRR", "MLP", "GP"
                 mlp_layers =       20,                                                 # number of layers for the MLP
                 augmentation =     True,                                               # load the data automatically
                 data_path =        "Perovskite_NC_synthesis_NH_240418.csv",
                ):
        

        # base parameters
        self.scoring_dim = None
        
        # training specific
        self.molecule_training_selection = molecule_training_selection
        self.expert = expert
        self.predictor = predictor
        self.mlp_layers = mlp_layers

        # dictionary of molecules and their geometry
        self.molecule_dictionary = self.get_molecule_dictionary()

        # data specific
        self.data_path =    data_path
        self.data =         self.read_data(molecule_training_selection)
        self.augmentation = augmentation

        # models
        if self.expert ==        "KRR":
            self.expert_models = self.train_expert_models()
        elif self.expert ==      "SIGMOID_FILTERS":
            self.expert_models = self.train_sigmoid_filters()
        elif self.expert ==      "GP":
            self.expert_models = self.train_expert_models_GP()

        ## - select between normal or popt based predictor training data - ##
        self.predictor_inputs, self.predictor_targets = self.generate_predictor_training_data()


        print(f"Predictor inputs: {len(self.predictor_inputs)} | Predictor targets: {len(self.predictor_targets)}")
        
        if self.predictor ==       "KRR":
            self.predictor_model = self.train_predictor_model_KRR(self.predictor_inputs, self.predictor_targets)
        elif self.predictor ==     "MLP":
            self.predictor_model = self.train_predictor_model_MLP(self.predictor_inputs, self.predictor_targets)
        elif self.predictor ==     "GP":
            self.predictor_model = self.train_predictor_model_GP(self.predictor_inputs, self.predictor_targets)
        else:
            raise ValueError("Predictor model not recognized")



    def read_data(self, molecule_training_selection) -> list:
        """
            Read in the data for a given molecule using the Datastructure class
        """

        data = []

        for molecule  in   molecule_training_selection:
            datastructure = Datastructure.Datastructure(synthesis_file_path= self.data_path,
                                                        target="PEAK_POS",
                                                        output_format="LIST",
                                                        wavelength_unit="NM",           # "NM", "EV"
                                                        monodispersity_only=True,       # only use monodisperse data
                                                        molecule=molecule,              # only use data for a specific molecule
                                                        )
            data_objects = datastructure.get_data()

            inputs   = [data["total_parameters"] for data in data_objects]  
            targets  = [data["y"] for data in data_objects]

            data.append({"molecule": molecule, "inputs": inputs, "targets": targets})
    
        return data



    def train_expert_models(self)       -> dict:
        """
            Train a set of expert models on the training data
        """

        expert_models = {}

        for dataset in self.data:
            inputs, targets = dataset["inputs"], dataset["targets"]
            
            if inputs == [] or targets == []:
                raise ValueError(f"Error: No data found for {dataset['molecule']}")

            #print(f"Optimizing hyperparameters for {dataset['molecule']} ...")

            #krr = self.optimize_hyperparameters(inputs, targets)

            krr = KernelRidge(alpha=0.001, gamma=0.001, kernel="laplacian")
            krr.fit(inputs, targets)

            expert_models[dataset["molecule"]] = krr
        
        return expert_models
    
    

    def train_expert_models_GP(self)    -> dict:
        """
            Train a set of expert models on the training data using Gaussian Processes
        """

        expert_models = {}

        for dataset in self.data:
            inputs, targets = dataset["inputs"], dataset["targets"]
            
            if inputs == [] or targets == []:
                raise ValueError(f"Error: No data found for {dataset['molecule']}")

            inputs =    np.reshape(inputs, (len(inputs), len(inputs[0])))
            targets =   np.reshape(targets, (len(targets), 1))

            kernel =    GPy.kern.Exponential(input_dim= len(inputs[0]))
            model =     GPy.models.GPRegression(inputs, targets, kernel)

            model.optimize(messages=True, max_f_eval=1000)

            expert_models[dataset["molecule"]] = model
        
        return expert_models



    def train_sigmoid_filters(self)     -> dict:
        """
            Train a set of sigmoid filters on the training data
        """
        
        sigmoid_filters = {}

        for dataset in self.data:
            inputs = [input[0] for input in dataset["inputs"]]
            targets = dataset["targets"]
            
            if inputs == [] or targets == []:
                print(f"Error: No data found for {dataset['molecule']}")
                return None
            
            # fit a Sigmoid filter to the data
            Sigmoid = lambda x, a, b, c, d: (a / (1 + np.exp(-b * (x - c)))) + d

            bounds = ([50, 0, -1, 460], [60, 30, 5, 463])           ### TODO: generalize the bounds, this is bad practice
            
            popt, _ = curve_fit(Sigmoid, inputs, targets, bounds=bounds, maxfev=10000)

            print(f"Optimized parameters for {dataset['molecule']}: {popt}")

            sigmoid_filters[dataset["molecule"]] = popt
        
        return sigmoid_filters



    def train_predictor_model_KRR(self, predictor_inputs, predictor_targets) -> KernelRidge:

        """
            Train the predictor model on the predictor inputs and targets
        """

        print("Training predictor model ...")
        

        ### - choose the best approach for hyperparameters - KRR - ###

        #predictor_model = self.optimize_hyperparameters(predictor_inputs, predictor_targets)
        predictor_model = KernelRidge(alpha=0.01, gamma=0.01, kernel="poly")

        ### ------------------------------------------------------- ###

        predictor_model.fit(predictor_inputs, predictor_targets)

        return predictor_model
    


    def train_predictor_model_MLP(self, predictor_inputs, predictor_targets) -> MLPRegressor:

        """
            Train the predictor model on the predictor inputs and targets
        """

        print("Training predictor model MLP ...")
        
        predictor_model = MLPRegressor(hidden_layer_sizes=(self.mlp_layers), solver='adam', alpha=0.0001, batch_size='auto', max_iter=1000, activation = "logistic")
        predictor_model.fit(predictor_inputs, predictor_targets)

        return predictor_model



    def train_predictor_model_GP(self, predictor_inputs, predictor_targets) -> GPy.models.GPRegression:

        """
            Train a Gaussian Process model on the predictor inputs and targets
        """

        print("Training predictor model GP ...")

        # reshape the inputs and targets for the GP model
        predictor_inputs =  np.reshape(predictor_inputs, (len(predictor_inputs), len(predictor_inputs[0])))
        predictor_targets = np.reshape(predictor_targets, (len(predictor_targets), 1))

        kernel = GPy.kern.Exponential(input_dim= len(predictor_inputs[0]))
        model = GPy.models.GPRegression(predictor_inputs, predictor_targets, kernel)

        model.optimize(messages=True, max_f_eval=1000)

        return model


    
    def forward_pass(self, input, molecule):

        """
            Predict the properties for synthesis parameters with a new molecule using the trained models
        """

        if molecule not in self.molecule_dictionary.keys():
            print(f"Error: Molecule {molecule} not in the scope of the model")
            return None

        predictor_input = []
        predictor_output = []


### ---------------- USE THIS  for the normal predictor training data ------------------ ###

        for basis_molecule in self.molecule_training_selection:
            if molecule != basis_molecule:
                
                scoring = self.scoring_function(molecule, basis_molecule)

                if self.expert ==   "KRR":
                    expert_opinion = self.expert_models[basis_molecule].predict([input])[0]

                elif self.expert == "SIGMOID_FILTERS":
                    popt = self.expert_models[basis_molecule]
                    expert_opinion = (popt[0] / (1 + np.exp(-popt[1] * (input[0] - popt[2])))) + popt[3]
                
                elif self.expert == "GP":
                    input = np.reshape(input, (1, len(input)))
                    expert_opinion = self.expert_models[basis_molecule].predict(input)[0][0][0]

                predictor_input.append( [expert_opinion] + scoring)

        for pred_input in predictor_input:
            if self.predictor == "GP":
                pred_input = np.reshape(pred_input, (1, len(pred_input)))
                
                predictor_output.append(self.predictor_model.predict(pred_input)[0][0])
            else:
                predictor_output.append( self.predictor_model.predict([pred_input]))
        return [np.mean(predictor_output)]
    



### -------------------------- HELPERS -------------------------- ###


    def optimize_hyperparameters(self, inputs, targets) -> KernelRidge:

        """
            Optimize the hyperparameters of the kernel ridge regression
        """

        krr = KernelRidge()

        parameters = {"alpha": [ 1e-6, 1e-4, 1e-2, 0.1,],
                      "gamma": [ 1e-6, 1e-5, 1e-4],
                      "kernel": ["laplacian", ]
                      }
        
        clf = GridSearchCV(krr, parameters, cv=5)
        clf.fit(inputs, targets)
        print(clf.best_params_)

        return clf.best_estimator_
    


    def scoring_function(self, molecule1, molecule2) -> list:

        """
            Scoring function for similarity between molecules
            TODO: add regularization for the scores, magnitudes are too different
        """

        ### ------------------ APPROACH 1: MANUALLY DEFINED SCORES ------------------ ###

        # group_score =       [0 if self.molecule_dictionary[molecule1]["group"] == self.molecule_dictionary[molecule2]["group"]           else 1                     ]
        # chainlength_score = [(self.molecule_dictionary[molecule1]["chainlength"] - self.molecule_dictionary[molecule2]["chainlength"])                              ]
        cycles_score =      [self.molecule_dictionary[molecule1]["cycles"]      - self.molecule_dictionary[molecule2]["cycles"]                                     ]
        hansen_score =      [(self.molecule_dictionary[molecule1]["hansen"]     - self.molecule_dictionary[molecule2]["hansen"])                                    ]
        # dipole_score =      [self.molecule_dictionary[molecule1]["dipole"]      - self.molecule_dictionary[molecule2]["dipole"]                                     ]
        relpol_score =      [self.molecule_dictionary[molecule1]["relative_polarity"] - self.molecule_dictionary[molecule2]["relative_polarity"]                    ]
        # diffusivity_score = [(self.molecule_dictionary[molecule1]["diffusivity"] - self.molecule_dictionary[molecule2]["diffusivity"]) * 10                         ]
        
        score =     relpol_score + hansen_score


        ### ------------------ APPROACH 2: SCORING DONE BY THE MODEL ----------------- ###

        #score = []
        #score += [self.molecule_dictionary[molecule1]["diffusivity"], self.molecule_dictionary[molecule2]["diffusivity"]]
        #score += [self.molecule_dictionary[molecule1]["hansen"], self.molecule_dictionary[molecule2]["hansen"]]
        #score += [self.molecule_dictionary[molecule1]["relative_polarity"], self.molecule_dictionary[molecule2]["relative_polarity"]]


        ## --------------------------------------------------------------------------- ###

        self.scoring_dim = len(score)
        return score
    


    def generate_predictor_training_data(self) -> list:

        """
            Generate scored inputs for the prediction model and corresponding targets
        """

        print("Generating predictor training data ...")

        prediction_inputs, prediction_targets = [], []

        for dataset in self.data:                                                   # iterate over the datasets (molecules)        

            molecule =          dataset["molecule"]
            inputs, targets =   dataset["inputs"], dataset["targets"]

            for input, target in zip(inputs, targets):                              # iterate over the data points

                predictor_input =   []
                predictor_target =  []

                for basis_molecule in self.molecule_training_selection:             # iterate over the basis molecules

                    #if molecule != basis_molecule:                                 # not sure ........???????????
                    score = self.scoring_function(molecule, basis_molecule)         # not efficient to calculate the score for each data point, change !!!!
                    
                    if self.expert == "KRR":
                        expert_opinion = self.expert_models[basis_molecule].predict([input])[0]

                    elif self.expert == "SIGMOID_FILTERS":
                        popt = self.expert_models[basis_molecule]
                        expert_opinion = (popt[0] / (1 + np.exp(-popt[1] * (input[0] - popt[2])))) + popt[3]

                    elif self.expert == "GP":
                        input = np.reshape(input, (1, len(input)))
                        expert_opinion = self.expert_models[basis_molecule].predict(input)[0][0][0]


                    predictor_input.append([expert_opinion] + score)
                    predictor_target.append(target)

                prediction_inputs  += predictor_input
                prediction_targets += predictor_target

        
        return prediction_inputs, prediction_targets



    def fit_final_sigmoid(self, molecule) -> list:

        """
            Fit the final sigmoid to the data from full model for unknown molecules
        """

        x_vec = np.linspace(0, 10, 300)
        y_vec = [self.forward_pass([x], molecule = molecule)[0] for x in x_vec]

        bounds = ([50, 8, -1, 460], [60, 30, 5, 463])
        Sigmoid = lambda x, a, b, c, d: (a / (1 + np.exp(-b * (x - c)))) + d

        popt, _ = curve_fit(Sigmoid, x_vec, y_vec, bounds=bounds, maxfev=10000)

        return popt

        

### ------------------------ DICTIONARY ------------------------- ###

    def get_molecule_dictionary(self) -> dict:

        """
            Get a dictionary of the molecules and their geometry
        """

        molecule_geometry = {
                        "Methanol":         {'group': [1,0], 'chainlength': 1, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 2.87, "dipole": 1.70, "relative_polarity" : 0.762, "hansen": 22.3,},
                        "Ethanol":          {'group': [1,0], 'chainlength': 2, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 1.15, "dipole": 1.68, "relative_polarity" : 0.654, "hansen": 19.4,},
                        "Propanol":         {'group': [1,0], 'chainlength': 3, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 0.75, "dipole": 1.65, "relative_polarity" : 0.617, "hansen": 17.4,},
                        "Isopropanol":      {'group': [1,0], 'chainlength': 3, 'cycles' : 0, 'group_pos': 1, 'diffusivity' : 0.56, "dipole": 1.58, "relative_polarity" : 0.546, "hansen": 16.4,},
                        "Butanol":          {'group': [1,0], 'chainlength': 4, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 0.47, "dipole": 1.66, "relative_polarity" : 0.586, "hansen": 15.8,},
                        "Hexanol":          {'group': [1,0], 'chainlength': 6, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 0.18, 'dipole': 1.60, "relative_polarity" : 0.559, "hansen": 12.5,},
                        "Octanol":          {'group': [1,0], 'chainlength': 8, 'cycles' : 0, 'group_pos': 0, 'diffusivity' : 0.07, 'dipole': 1.68, "relative_polarity" : 0.537, "hansen": 11.2,},
                        "Acetone":          {'group': [0,1], 'chainlength': 3, 'cycles' : 0, 'group_pos': 1,                       "dipole": 2.86, "relative_polarity" : 0.355, "hansen": 7.0,},
                        "Butanone":         {'group': [0,1], 'chainlength': 4, 'cycles' : 0, 'group_pos': 1,                       "dipole": 2.78, "relative_polarity" : 0.327, "hansen": 5.1,},
                        "Cyclopentanone":   {'group': [0,1], 'chainlength': 5, 'cycles' : 1, 'group_pos': 0,                       "dipole": 3.30, "relative_polarity" : 0.269, "hansen": 5.2,},
        }

        return molecule_geometry




### ------------------------ PLOTTING ------------------------- ###

    def viz_experts(self):

        """
            Visualize the expert models
        """
        
        x_vec = np.linspace(0, 10, 300)
        
        for expert in self.expert_models.keys():

            if self.expert == "KRR":
                y_vec = [self.expert_models[expert].predict([[x]])[0] for x in x_vec]

            elif self.expert == "SIGMOID_FILTERS":
                popt = self.expert_models[expert]
                y_vec  = [(popt[0] / (1 + np.exp(-popt[1] * (x - popt[2])))) + popt[3] for x in x_vec]

            elif self.expert == "GP":
                y_vec = [self.expert_models[expert].predict(np.reshape([x], (1, 1)))[0][0][0] for x in x_vec]
                print(y_vec)

            #inputs =  [input[0] for input in self.data[self.molecule_training_selection.index(expert)]["inputs"]]
            #targets = [target for target in self.data[self.molecule_training_selection.index(expert)]["targets"]]

            fig = go.Figure()

            fig.add_trace(go.Scatter(x= x_vec, y= y_vec, mode='lines', name= f"{expert}_model"))
            #fig.add_trace(go.Scatter(x= inputs, y= targets, mode='markers', name= f"{expert}_data"))
            
            fig.show()
