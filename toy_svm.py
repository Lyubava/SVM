#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
Created on April 8, 2016
"""


import matplotlib.pyplot as plt
import numpy
import os

from sgd import SGD


class ToyLoader(object):
    """
    Loads and analyses toy dataset
    """
    def __init__(self, path_to_data, path_to_targets):
        self.path_to_data = path_to_data
        self.path_to_targets = path_to_targets
        self.features = None
        self.targets = None

    def load_data(self):
        with open(self.path_to_data, "r") as fin:
            features = [
                [float(elem) for elem in line.strip().split(",")]
                for line in fin]
        with open(self.path_to_targets, "r") as fin:
            targets = [int(line.strip()) for line in fin]
        assert len(features) == len(targets)
        self.features = numpy.array(features, dtype=numpy.float32)
        self.targets = numpy.array(targets, dtype=numpy.int8)
        return self.features, self.targets

    def analyse_data(self):
        def get_min_max_mean(array_name):
            data = getattr(self, array_name)
            max_elem = data.max()
            min_elem = data.min()
            mean_elem = data.mean()
            print(
                "%s: max elem: %s, min elem: %s, mean elem: %s" %
                (array_name, max_elem, min_elem, mean_elem))
        get_min_max_mean("features")
        get_min_max_mean("targets")


class ToyWorkflow(object):
    def __init__(self, base_path, batch_size):
        self.path_to_data = os.path.join(base_path, "features.txt")
        self.path_to_targets = os.path.join(base_path, "targets.txt")
        self.accuracy = None
        self.loss = None
        self.minibatch_size = batch_size

    def run(self):
        loader = ToyLoader(
            path_to_data=self.path_to_data,
            path_to_targets=self.path_to_targets)
        input_data, input_targets = loader.load_data()
        input_targets = input_targets.reshape(input_targets.shape[0], 1)
        loader.analyse_data()
        sgd = SGD(
            input_data, input_targets, loss_type="hinge",
            regularization_type="l2", learning_rate=0.0001, lambda_coeff=1,
            max_epochs=100, max_failed_iterations=0,
            minibatch_size=self.minibatch_size,
            n_valid_samples=1000, initialization="zero",
            stop_condition="accuracy", model="linear_svm")
        sgd.run()
        self.accuracy, self.loss = sgd.accuracy_per_epoch, sgd.loss_per_epoch
        return sgd.best_weights, sgd.best_bias

    def plotting(self, name_to_plot):
        data_to_plot = getattr(self, name_to_plot)
        epochs = list(range(len(data_to_plot["train"])))
        fig = plt.figure()
        title = " minibatch size: " + str(self.minibatch_size)
        for set_name in ("train", "valid"):
            plt.plot(
                epochs, data_to_plot[set_name], label=set_name)
            plt.legend()
        fig.suptitle(title, fontsize=20)
        plt.xlabel("epochs", fontsize=16)
        plt.ylabel(name_to_plot, fontsize=18)
        plt.show()


if __name__ == "__main__":
    input_path = "data"
    workflow = ToyWorkflow(input_path, 100)
    best_weights, best_bias = workflow.run()
    workflow.plotting("accuracy")
    workflow.plotting("loss")
