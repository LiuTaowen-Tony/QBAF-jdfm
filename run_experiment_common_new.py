import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from genetic_algorithm.genetic_generial import GeneralGeneticAlgorithm
import json
from datasets.mushrooms import load_mushroom
from datasets.adult import load_adult
from datasets.iris import load_iris
from genetic_algorithm.DAGBAG import DAGBAG
from genetic_algorithm.GBAG import GBAG
from genetic_algorithm.JASGBAG import JASGBAG
from genetic_algorithm.JASDAGBAG import JASDAGBAG
from genetic_algorithm.MultiLayerGBAG import MultiGBAG

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cpu")

def parse_config(dataset, num_of_jas, which_size, has_direct, relative_sparsity, is_fuzzy):
    base_config =  {

        'number_runs': 10, 

        'population_size': 20, 'number_generations': 10, 

        'learning_rate': 0.01, 'number_epochs': 300, 
        
        'hidden_size': 12, 'number_connections1': 6, 'number_connections2': 6, 
        
        'lambda': 0.05, 
        
        'crossover_rate': 0.9, 'mutation_rate': 0.001, 
        'patience_ES': 5, 'tolerance_ES': 3e-3, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4}

    if relative_sparsity:
        base_config['relative_sparsity'] = True
        if dataset == 'adult':
            'lambda: 0.02'
    if dataset == 'iris':
        base_config['tolerance_ES'] = 1e-4

    base_config['is_fuzzy'] = is_fuzzy
    if num_of_jas != 0 and has_direct:
        model_name = 'JASDAGBAG'
    elif num_of_jas != 0:
        model_name = 'JASGBAG'
    elif has_direct:
        model_name = 'DAGBAG'
    else:
        model_name = 'GBAG'

    base_config['model_name'] = model_name
    base_config['dataset'] = dataset

    if dataset == 'iris':
        base_config['population_size'] = 20
    elif dataset == 'adult':
        base_config['population_size'] = 50
    elif dataset == 'mushroom':
        base_config['population_size'] = 30

    neuron_configs = {
        1: {
            'hidden_size': 12,
            'number_connections1': 10,
            'number_connections2': 6,
        },
        2: {
            'hidden_size': 12,
            'number_connections1': 6,
            'number_connections2': 6,
        },
        3: {
            'hidden_size': 8,
            'number_connections1': 4,
            'number_connections2': 4,
        },
        4: {
            'hidden_size': 6,
            'number_connections1': 4,
            'number_connections2':2,
        },
        5: {
            'hidden_size': 4,
            'number_connections1': 4,
            'number_connections2':2,
        },
    }

    neuron_config = neuron_configs[which_size]

    if num_of_jas != 0:
        neuron_config['number_connections1'] -= num_of_jas
        neuron_config.update(
         {
         "joint_connections_size1": num_of_jas,
         "joint_connections_size2": 1,
         "joint_connections_input_num1": 4,
         "joint_connections_input_num2": 2,
         }
        )

    if has_direct:
        neuron_config['number_connections_skip'] = 1


    fuzzy_str = 'fuzzy' if is_fuzzy else 'nf'
    sparsity_str = 'sp' if relative_sparsity else 'nsp'
    config_name = f'{dataset}_{fuzzy_str}_{model_name}_s{which_size}_j{num_of_jas}_{sparsity_str}_new'
    base_config['config_name'] = config_name

    base_config.update(neuron_config)
    return base_config


# def parse_config(dataset, num_of_jas, which_size, has_direct, relative_sparsity, is_fuzzy):
parameters = parse_config(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
             sys.argv[4] == 'direct', sys.argv[5] == 'sp',
             sys.argv[6] == 'fuzzy')

print("Running with parameters:" + str(parameters), flush=True)

dataset = parameters["dataset"]
if dataset == "adult":
    X, y, inputs, label = load_adult(is_fuzzy=parameters["is_fuzzy"])
elif dataset == "mushroom":
    X, y, inputs, label = load_mushroom(is_fuzzy=parameters["is_fuzzy"])
elif dataset == "iris":
    X, y, inputs, label = load_iris(is_fuzzy=parameters["is_fuzzy"])

model_name = parameters["model_name"]
if model_name == "GBAG":
    parameters["algo_class"] = GBAG
elif model_name == "DAGBAG":
    parameters["algo_class"] = DAGBAG
elif model_name == "JASGBAG":
    parameters["algo_class"] = JASGBAG
elif model_name == "JASDAGBAG":
    parameters["algo_class"] = JASDAGBAG
elif model_name == "MULTIGBAG":
    parameters["algo_class"] = MultiGBAG

params = parameters
torch.manual_seed(2021)
np.random.seed(2021)  # scikit-learn also uses numpy random seed
for run in range(params["number_runs"]):
    X = torch.tensor(X).float()
    y = torch.tensor(y)
    y = y.float() if dataset == "iris" else y.long()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.125, stratify=y_tr
    )

    criterion = torch.nn.CrossEntropyLoss()
    model = GeneralGeneticAlgorithm(
        input_size=X_tr.shape[1],
        output_size= 3 if dataset == "iris" else 2,
        selection_method="tournament_selection",
        crossover_method="two_point_crossover",
        mutation_method="flip_mutation",
        params=params,
        loss_function=criterion,
        x_train=X_tr,
        y_train=y_tr,
        x_val=X_val,
        y_val=y_val,
        x_test=X_te,
        y_test=y_te,
    )
    model.run(
        input_labels=inputs,
        class_labels=label,
        )

