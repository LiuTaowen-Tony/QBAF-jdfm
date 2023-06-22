#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re


def parse_config(file_name):

    with open(file_name, 'r') as file:
        file.readline()
        text = file.readline()

    end = text.find("}\"") + 1

    text = text[1:end]

    # text = "{'number_runs': 10, 'population_size': 20, 'number_generations': 10, 'learning_rate': 0.01, 'number_epochs': 300, 'hidden_size': 4, 'number_connections1': 4, 'number_connections2': 2, 'lambda': 0.05, 'crossover_rate': 0.9, 'mutation_rate': 0.001, 'patience_ES': 5, 'tolerance_ES': 0.0001, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 0.0001, 'is_fuzzy': True, 'model_name': 'GBAG', 'dataset': 'iris', 'config_name': 'iris_fuzzy_GBAG_s5_j0_nsp_new', 'algo_class': <class 'genetic_algorithm.GBAG.GBAG'>, 'input_size': 12, 'output_size': 3}"

    # Replace single quotes with double quotes
    text = text.replace("'", "\"")
    text = text.replace("_", "-")
    text = text.replace("True", '"true"')
    text = text.replace("False", '"false"')

    # Replace class reference with a placeholder
    text = re.sub(r"<([^<]+)>", '""', text)

    # Parse the text into a dictionary
    data = json.loads(text)

    # Convert the dictionary to a table
    del data["algo-class"]
    config_name = data["config-name"]
    print(config_name)
    
    with open("config_table.tex", "a") as file:
        if "fuzzy" in config_name:
            print("Fuzzy", file=file, end = " ")
        if "JASDAGBAG" in config_name:
            print("Joint and Direct", file=file)
        elif "JASGBAG" in config_name:
            print("Joint", file=file)
        elif "DAGBAG" in config_name:
            print("Direct", file=file)
        elif "MULTI" in config_name:
            print("Direct", file=file)
        else:
            print("Baseline", file=file)

        


    del data["config-name"]
    with open("config_table.tex", "a") as file:
        print(data["dataset"], file=file)
        print("\\begin{itemize}", file=file)
        for key, value in data.items():
            print("\\item", key, value, file=file)
        print("\\end{itemize}", file=file)
        print("", file=file)

while True:
    
    file_name = input("Enter the name of the config file: ")

    parse_config(file_name)
