import numpy as np
import torch
import csv
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import typing
from typing import *
import math
import time

from genetic_algorithm.SparseAlgo import SparseAlgo

from genetic_algorithm.operators.crossover import (
    SinglePointCrossover,
    TwoPointCrossover,
)
from genetic_algorithm.operators.mutation import (
    FlipMutation,
    SwapMutationBetweenChromosomes,
)
from genetic_algorithm.operators.selection import (
    RankSelection,
    RouletteWheelSelection,
    TournamentSelection,
)

from genetic_algorithm.utils import metrics
from genetic_algorithm.utils.plots import plot_fitness, plot_loss, plot_conf_matrix

from datetime import datetime
now = datetime.now()
start_time = now.strftime("%d,%H:%M:%S")

def print(*args, **kwargs):
    """Prints the arguments."""
    now = datetime.now()
    current_time = now.strftime("%d,%H:%M:%S")

    __builtins__["print"](current_time, *args, **kwargs, flush=True)


selection_operators = {
    "roulette_wheel_selection": RouletteWheelSelection,
    "rank_selection": RankSelection,
    "tournament_selection": TournamentSelection,
}
crossover_operators = {
    "one_point_crossover": SinglePointCrossover,
    "two_point_crossover": TwoPointCrossover,
}
mutation_operators = {
    "flip_mutation": FlipMutation,
    "swap_mutation": SwapMutationBetweenChromosomes,
}


class GeneralGeneticAlgorithm():
    """Implementation of a genetic algorithm to evolve the structure of (sparse) multilayer perceptrons / GBAGs

    Parameters
    ----------
    input_size : number of input features

    output_size : number of classes

    selection method: method for selection operator
                    {'roulette_wheel_selection', 'tournament_selection', 'rank_selection'}

    crossover_method : method for crossover operator
                    {'one_point_crossover', 'two_point_crossover'}

    mutation_method : method for mutation operator
                    {'flip_mutation', 'swap_mutation'}

    params : dict containing necessary parameters, e.g.

            {'number_runs': 10, 'population_size': 20, 'number_generations': 20, 'crossover_rate': 0.9,
             'mutation_rate': 0.001, 'learning_rate': 3e-2, 'number_epochs': 3000, 'hidden_size': 12,
             'number_connections1': 12, 'number_connections2': 6, 'lambda': 0.1, 'patience_ES': 5,
             'tolerance_ES': 1e-4, 'elitist_pct': 0.1, 'patience_GA': 5, 'tolerance_GA': 1e-4}

             ES: Early Stopping
             GA: Genetic Algorithm

    loss_function : loss function for parameter and structure learning

    show_graph : show argumentation classifier graph
                {'True', 'False'}, default='False'

    show_plots : show plots for mean and best fitness, training and validation loss and confusion matrix
                {'True', 'False'}, default='False'

    """

    def __init__(
        self,
        input_size,
        output_size,
        selection_method,
        crossover_method,
        mutation_method,
        params,
        loss_function,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        show_graph=False,
        show_plots=False,
    ):
        self.input_size = input_size
        self.output_size = output_size
        try:
            self.selection_operator = selection_operators[selection_method](
                num_parents=int(0.5 * params["population_size"])
            )
            self.crossover_operator = crossover_operators[crossover_method](
                params["crossover_rate"],
                num_offspring=int(
                    (1 - params["elitist_pct"]) * params["population_size"]
                ),
            )
            self.mutation_operator = mutation_operators[mutation_method](
                params["mutation_rate"]
            )
        except KeyError as e:
            raise NotImplementedError(
                f"Got unknown method {e} for one of the operators."
            )
        self.params = params
        self.loss_function = loss_function
        self.show_graph = show_graph
        self.show_plots = show_plots
        self.population = None        
        train_ds = TensorDataset(x_train, y_train)
        self.train_dl = DataLoader(train_ds, batch_size=25, shuffle=True)
        self.x_val = x_val
        self.y_val = y_val
        self.X_te = x_test
        self.y_te = y_test
        self.params["input_size"] = input_size
        self.params["output_size"] = output_size
        self.mean_num_connections = 0


    def encode(self, parents: List[SparseAlgo]) -> List[List[torch.Tensor]]:
        """Encodes the structure of the graph as a bit string for genetic algorithm.
 
        The rows of the connectivity matrix are concatenated.
        """
        return [p.get_mask_matrix_encoding() for p in parents]

    def decode(
            self, 
            flattened_mask_matrices_list: List[torch.Tensor]
        ) -> List[SparseAlgo]:
        cls = self.params["algo_class"]
        return [cls.from_flattened_mask_matrices(
                params=self.params,
                mask_matrix_encoding=i) 
            for i in flattened_mask_matrices_list]

    def create_population(self):
        """Creates the initial population."""
        cls = self.params["algo_class"]
        self.population = []
        for _ in range(self.params["population_size"]):
            self.population.append(
                cls.random_connectivity_init( self.params)
            )

        # joint_feature_proportion = self.params["joint_feature_proportion"]
        # normal_feature_proportion = 1 - joint_feature_proportion
        # n_connections_input_hidden = math.ceil(self.params["n_connections_input_hidden"] * normal_feature_proportion)
        # n_connections_hidden_output = math.ceil(self.params["n_connections_hidden_output"] * normal_feature_proportion)
        # n_connections_jointly_input_hidden = math.floor(self.params["n_connections_input_hidden"] * joint_feature_proportion)
        # n_connections_jointly_hidden_output = math.floor(self.params["n_connections_hidden_output"] * joint_feature_proportion)
        # self.population = []
        # for _ in range(self.params["population_size"]):
        #     self.population.append(
        #         SparseAlgo.random_connectivity_init(
        #             input_size=self.input_size,
        #             hidden_size=self.params["hidden_neuron_size"],
        #             output_size=self.output_size,
        #             n_connections_input_hidden=n_connections_input_hidden,
        #             n_connections_hidden_output=n_connections_hidden_output,
        #             n_connections_jointly_input_hidden=n_connections_jointly_input_hidden,
        #             n_connections_jointly_hidden_output=n_connections_jointly_hidden_output,
        #         )
        #     )

    def create_new_generation(self, elitist: List[SparseAlgo], mutated_encodings: List[torch.Tensor]):
        """Creates a new generation.

        Generation is created from best individuals (elitism) and mutated offspring.

        Args:
            elitist: A list containing the best individuals.
            mutation_offspring: The first half of the mutated offspring.
            mutation_offspring2: The second half of the mutated offspring.
        """
        
        cls = self.params["algo_class"]
        self.population = []
        for i in elitist:
            # if i.total_num_conn() > self.mean_num_connections + 2:
            #     i = cls.random_connectivity_init(self.params)
            self.population.append(i)

        # reshape connectivity matrices and create new G-BAGs (offspring) given the new connectivity matrix
        new_generation_size = math.ceil((1 - self.params["elitist_pct"]) * self.params["population_size"])
        i = 0
        while i < new_generation_size:
            child = cls.from_mask_matrix_encoding(
                params=self.params,
                mask_matrix_encoding=mutated_encodings[i])
            # if child.total_num_conn() > self.mean_num_connections + 1:
            #     child = cls.random_connectivity_init(self.params)
            self.population.append(child)
            i += 1

    def fitness(
        self,
        population: List[SparseAlgo],
        opt_func=torch.optim.Adam,
        minibatch=True,
    ):
        """Gets the fitness/loss of each individual."""

        sparsities = [
            individual.find_sparsity() for individual in population
        ]
        min_sparsity = min(sparsities)
        max_sparsity = max(sparsities)

        for i, individual in enumerate(population):
            optimizer = opt_func(
                individual.parameters(), lr=self.params["learning_rate"]
            )
            individual.fitness = torch.autograd.Variable(
                torch.tensor(np.inf, dtype=torch.float)
            )
            train_loss = []
            validation_loss = []
            best_score = None
            count = 0
            for epoch in range(self.params["number_epochs"]):
                batch_loss = 0.0
                batch_accuracy = 0.0
                for nb, (x_batch, y_batch) in enumerate(self.train_dl):
                    # print(nb)
                    optimizer.zero_grad()
                    y_pred_train = individual(x_batch)
                    loss = self.loss_function(y_pred_train, y_batch)
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                    batch_accuracy += metrics.accuracy(y_pred_train, y_batch).item()
                train_loss.append(batch_loss / (nb + 1))
                accuracy = batch_accuracy / (nb + 1)
                relative_sparsity = (individual.find_sparsity() - min_sparsity) / (
                    max_sparsity - min_sparsity + 1e-8
                )
                if "relative_sparsity" in self.params and self.params["relative_sparsity"]:
                    sp_for_fit = relative_sparsity
                else:
                    sp_for_fit = individual.find_sparsity()
                individual.fitness = ((1 - self.params["lambda"]) * accuracy 
                                      + self.params["lambda"] * (sp_for_fit))
                individual.accuracy = accuracy
                individual.training_loss = train_loss
                # Early Stopping
                with torch.no_grad():
                    individual.eval()
                    y_pred_val = individual(self.x_val)
                    val_loss = self.loss_function(y_pred_val, self.y_val)
                    validation_loss.append(val_loss.item())
                score = -val_loss.item()
                if best_score is None:
                    best_score = score
                elif score < best_score + self.params["tolerance_ES"]:
                    count += 1
                else:
                    best_score = score
                    count = 0
                if count >= self.params["patience_ES"]:
                    break
            individual.val_loss = validation_loss

    def run(
        self,
        input_labels,
        class_labels,
    ):
        """Runs the genetic algorithm for a given configuration.

        inputs:
                - training (tr), validation (val) and test (te) data
                - input labels (array containing labels of input features)
                - class labels (array containing class labels)
                - file name for csv to save results
        """
        self.create_population()
        print("Initial population created")
        self.fitness(self.population)

        best_fitness = []
        mean_fitness = []
        count = 0
        best_score = None
        num_of_elite = int(self.params["elitist_pct"] * self.params["population_size"])

        for g in range(self.params["number_generations"]):
            print(f"Generation {g}")

            n_conns = [indiv.total_num_conn() for indiv in self.population]
            hist_count, hist_bins = np.histogram(n_conns, bins=10)
            print("histogram of n_conns values", hist_count, hist_bins)
            fitness = [indiv.fitness for indiv in self.population]
            accuracy = [indiv.accuracy for indiv in self.population]
            hist_count, hist_bins = np.histogram(accuracy, bins=10)
            print("histogram of accuracy values", hist_count, hist_bins)
            hist_count, hist_bins = np.histogram(fitness, bins=10)
            print("histogram of fitness values", hist_count, hist_bins)
            best_fitness.append(max(fitness))
            mean_fitness.append(np.mean(fitness))
            print("mean_connections", np.mean([indiv.total_num_conn() for indiv in self.population]))
            self.mean_num_connections = np.mean([indiv.total_num_conn() for indiv in self.population])

            # stop genetic algorithm if accuracy does not increase for a certain number of generations
            if best_score is None:
                best_score = best_fitness[g]
            elif best_fitness[g] - best_score < self.params["tolerance_GA"]:
                print(f"hit tolerance GA cur: {best_fitness[g]} best: {best_score} count: {count}")
                count += 1
            else:
                best_score = best_fitness[g]
                count = 0
            if count >= self.params["patience_GA"]:
                break

            print("selecting", end = " ")
            # elitism: pass certain percentage of best individuals directly to next generation
            fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
            fitness_sorted.reverse()  # sort in descending order
            elitist = []  # fitness proportionate selection
            for index in sorted(fitness_sorted[:num_of_elite]):
                elitist.append(self.population[index])

            # selection
            parents = self.selection_operator.select(self.population)

            # encoding of chromosomes
            encoded_mask_matrices_list = self.encode(parents)

            for i in encoded_mask_matrices_list:
                for j in i:
                    j_ = torch.tensor(j)
                    for k in j_.flatten():
                        assert k.round().to(torch.long) in [0, 1]

            print("crossovering", end = " ")
            # crossover
            encoded_mask_matrices_list_T = list(zip(*encoded_mask_matrices_list))
            crossover_offspring_T = []
            for i in encoded_mask_matrices_list_T:
                shape = i[0].shape
                for j in i:
                    assert j.shape == shape
                flattend_i = [j.flatten() for j in i]
                stacked_i = torch.stack(flattend_i)
                crossovered_offspring_i = self.crossover_operator.crossover(stacked_i)
                restored_i = [j.reshape(shape) for j in crossovered_offspring_i]
                crossover_offspring_T.append(restored_i)
                
            for i in crossover_offspring_T:
                for j in i:
                    j_ = torch.tensor(j)
                    for k in j_.flatten():
                        assert k.round().to(torch.long) in [0, 1]

            print("mutating", end = " ")
            # mutation
            mutation_offspring_T = []
            for i in crossover_offspring_T:
                shape = i[0].shape
                for j in i:
                    assert j.shape == shape
                flattend_i = [torch.tensor(j.flatten()) for j in i]
                stacked_i = torch.stack(flattend_i)
                mutation_offspring_i = self.mutation_operator.mutate(stacked_i)
                restored_i = [j.reshape(shape) for j in mutation_offspring_i]
                mutation_offspring_T.append(torch.tensor(restored_i))
            mutation_offspring = list(zip(*mutation_offspring_T))
        
            for i in mutation_offspring_T:
                for j in i:
                    j_ = torch.tensor(j)
                    for k in j_.flatten():
                        assert k.round().to(torch.long) in [0, 1]

            # form new generation
            self.create_new_generation(elitist, mutation_offspring)

            # evaluate fitness of new population
            self.fitness( self.population[ num_of_elite : ],)

            print("Generation {} finished".format(g + 1))
            print(f"Best fitness: {best_fitness[-1]}")

        # select best individual and return results
        acc = [indiv.accuracy for indiv in self.population]
        fitness = [indiv.fitness for indiv in self.population]
        idx = np.argmax(fitness)
        clf = self.population[idx]
        best_fitness.append(clf.fitness)
        mean_fitness.append(np.mean(fitness))
        training_accuracy = clf.accuracy
        print(
            "Best individual - accuracy on training data: {:.4}".format(
                training_accuracy
            )
        )
        print("Mean accuracy on training data: {:.4}".format(np.mean(acc)))

        # evaluate best model on test data
        with torch.no_grad():
            clf.eval()
            y_pred = clf(self.X_te)
            test_loss = self.loss_function(y_pred, self.y_te)
            test_accuracy = metrics.accuracy(y_pred, self.y_te)
            print("Best individual - loss on test data: {:.4}".format(test_loss))
            print(
                "Best individual - accuracy on test data: {:.4}".format(test_accuracy)
            )

            classes = torch.argmax(y_pred, dim=1)
            if len(self.y_te.shape) > 1:
                labels = torch.argmax(self.y_te, dim=1)
            else:
                labels = self.y_te

            precision = precision_score(labels, classes, average="macro")
            recall = recall_score(labels, classes, average="macro")
            f1 = f1_score(labels, classes, average="macro")

        # remove not meaningful connections
        # rem, i_ind = remove_connections(clf)
        num_connections = clf.reduced_num_conn()
        
        print("Best individual - number of connections: {}".format(num_connections))

        # visualizations (default: not shown, set 'show_plots' to 'True' otherwise)
        #  if self.show_plots:
            #  plot_fitness(best_fitness, mean_fitness)
            #  plot_loss(clf.training_loss, clf.val_loss)
            #  plot_conf_matrix(labels, classes, class_labels)

        # draw argumentation classifier (default: not shown, set 'show_graph' to 'True' otherwise)
        #  if self.show_graph:
            #  inp_used = []
            #  for inn in i_ind:
                #  inp_used.append(input_labels[inn.item()])
#
            #  ind1 = rem.sparse_linear1.weight._indices()
            #  val1 = rem.sparse_linear1.weight._values()
            #  s1 = rem.sparse_linear1.out_features
            #  s2 = rem.sparse_linear1.in_features
            #  weights1 = torch.sparse.FloatTensor(
                #  ind1, val1, torch.Size([s1, s2])
            #  ).to_dense()
#
            #  ind2 = rem.sparse_linear2.weight._indices()
            #  val2 = rem.sparse_linear2.weight._values()
            #  s3 = rem.sparse_linear2.out_features
            #  s4 = rem.sparse_linear2.in_features
            #  weights2 = torch.sparse.FloatTensor(
                #  ind2, val2, torch.Size([s3, s4])
            #  ).to_dense()
#
            #  number_of_neurons_in_widest_layer = max(
                #  rem.sparse_linear1.in_features, rem.sparse_linear2.out_features
            #  )
            #  network = NeuralNetwork(number_of_neurons_in_widest_layer)
            #  network.add_layer(s2, weights1)
            #  network.add_layer(s4, weights2)
            #  network.add_layer(s3)
            #  network.draw(inp_used, class_labels)

        # write results to csv file
        import socket
        import os
        host_name = socket.gethostname()
        dataset_name = self.params["dataset"]
        config_name = self.params["config_name"]

        file_name = f"{dataset_name}_{config_name}_{start_time}_{host_name}.csv"
        experiment_name = config_name[str(config_name).find('_'):]
        folder_name = f"results/{experiment_name}"
        
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if not os.path.exists("describe"):
            os.makedirs("describe")
        file_name_describe = file_name[:-4] + "_describe.txt"
        import json
        with open(f"describe/{file_name_describe}", "a") as file:
            json.dump(clf.describe(), file, indent=4)
        if not os.path.isfile(f"{folder_name}/{file_name}"):
            with open(f"{folder_name}/{file_name}", "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "params",
                        "num_connections",
                        "training_accuracy",
                        "test_accuracy",
                        "recall",
                        "precision",
                        "f1",
                        "generation"
                    ]
                )
        with open(f"{folder_name}/{file_name}", "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    self.params,
                    num_connections,
                    round(training_accuracy, 4),
                    round(test_accuracy.item(), 4),
                    round(recall, 4),
                    round(precision, 4),
                    round(f1, 4),
                    g
                ]
            )
