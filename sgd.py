# -*-coding: utf-8 -*-
"""
Created on April 8, 2016
"""

import copy
import numpy
import time

from loss_function import LossFunctionsRegistry
from regularization_functions import RegularizationRegistry
from svm import ModelsRegistry


class SGD(object):
    def __init__(
            self, input_array, target_array, loss_type="hinge",
            regularization_type="l2", learning_rate=0.1, lambda_coeff=0.05,
            max_epochs=100, max_failed_iterations=10, minibatch_size=1,
            n_valid_samples=1000, initialization="zero", random_seed=42,
            stop_condition="loss", model="linear_svm"):

        # TODO: add shuffle dataset

        self.input = input_array
        self.target = target_array
        self.max_minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.lambda_coeff = lambda_coeff
        self.max_epochs = max_epochs
        self.max_failed_iterations = max_failed_iterations
        self.loss_type = loss_type
        self.regularization_type = regularization_type
        self.model_name = model
        self.max_valid_accuracy = 0.0
        self.epoch = 0
        self.failed_iterations = 0
        self.random_seed = random_seed
        self.initialization = initialization
        self.stop_condition = stop_condition
        self.prev_loss_valid = numpy.iinfo(numpy.uint32).max
        self.accuracy_per_epoch = {}
        self.loss_per_epoch = {}

        for set_name in ("valid", "train"):
            for name in ("_input", "_target", "batches_n"):
                setattr(self, set_name + name, None)
            self.accuracy_per_epoch[set_name] = [self.max_valid_accuracy]
            self.loss_per_epoch[set_name] = [self.prev_loss_valid]

        for attr_name in (
                "prediction", "weights", "bias", "target_train_minibatch",
                "minibatch", "loss_function", "regularization",
                "minibatch_size", "best_weights", "best_bias",
                "current_loss_valid", "current_loss_train"):
            setattr(self, attr_name, None)

        self.n_valid_samples = n_valid_samples
        self.n_total_samples, self.n_features = self.input.shape
        self.n_train_samples = self.n_total_samples - self.n_valid_samples
        self.n_valid_samples = self.n_valid_samples

        self.split_data_to_train_and_validation()
        self.weights, self.bias = getattr(
            self, "initialize_weights_" + self.initialization)()

        self.loss_function = LossFunctionsRegistry.loss_functions[
            self.loss_type]
        self.regularization = RegularizationRegistry.regularization[
            self.regularization_type]
        self.model = ModelsRegistry.models[self.model_name]

    @staticmethod
    def get_max_min_value(data):
        return data.max(), data.min()

    def get_batches_number(self, n_samples):
        """
        Gets number of minibatches fron number of samples and minibatch size
        """
        return int(numpy.ceil(n_samples / self.max_minibatch_size))

    def split_data_to_train_and_validation(self):
        """
        Splits data and targets to train and validation sets
        """
        self.train_batches_n = self.get_batches_number(self.n_train_samples)
        self.train_input = self.input[:self.n_train_samples]
        self.train_target = self.target[:self.n_train_samples]
        self.valid_batches_n = self.get_batches_number(self.n_valid_samples)
        self.valid_input = self.input[
            self.n_train_samples: self.n_total_samples]
        self.valid_target = self.target[
            self.n_train_samples: self.n_total_samples]
        assert self.train_input.shape == (
            self.n_train_samples, self.n_features)
        assert self.valid_input.shape == (
            self.n_valid_samples, self.n_features)

    def initialize_weights_zero(self):
        """
        Set weights and bias to zero
        """
        weights = numpy.zeros(
            (self.n_features, 1), dtype=numpy.float32)
        bias = 0
        return weights, bias

    def initialize_weights_random(self):
        """
        Set weights and bias to random
        """
        numpy.random.seed(self.random_seed)
        max_value, min_value = self.get_max_min_value(self.train_input)
        weights = numpy.random.uniform(
            min_value, max_value, (self.n_features, 1))
        bias = numpy.random.uniform(min_value, max_value)
        return weights, bias

    def get_minibatches(self, minibatch_index, set_name):
        """
        Finds current minibatch_size, minibatch and its targets
        on train/validation
        """
        batches_number = getattr(self, set_name + "_batches_n")
        samples_number = getattr(self, "n_%s_samples" % set_name)
        input_set = getattr(self, set_name + "_input")
        target_set = getattr(self, set_name + "_target")

        if minibatch_index != batches_number - 1:
            self.minibatch_size = self.max_minibatch_size
        else:
            self.minibatch_size = (
                samples_number - (batches_number - 1) *
                self.max_minibatch_size)
        return (
            input_set[minibatch_index:minibatch_index + self.minibatch_size],
            target_set[minibatch_index:minibatch_index + self.minibatch_size])

    def train_epoch(self):
        """
        Trains training dataset one time
        """
        train_time_start = time.time()
        good_train_samples = 0
        train_loss = []
        for minibatch_index in range(self.train_batches_n):
            self.minibatch, self.target_train_minibatch = \
                self.get_minibatches(minibatch_index, "train")

            self.prediction = numpy.dot(
                self.minibatch, self.weights) - self.bias

            loss_function = self.loss_function(
                target=self.target_train_minibatch, prediction=self.prediction)

            regularization = self.regularization(
                weights=self.weights,
                lambda_coeff=self.lambda_coeff)

            model = self.model(
                self.n_features, self.minibatch_size, self.minibatch)

            # Need to minimize: F = 1/n * sum(E(y(w, b))) + R(w, b),
            # where y(w, b) - model and E(y) - loss function
            # Need to find partial derivative on weights and bias:
            # dF/dw = 1/n * sum(dE/dy * dy/dw) + dR/dw
            # dF/db = 1/n * sum(dE/dy * dy/db) + dR/db

            # Find sum(dE/dy * dy/dw), where dE/dy - loss_grad parameter,
            # dE/dy calculates in loss_function.get_weights_loss_grad(),
            # dy/dw calculates in model.get_sum_grad_weights:

            loss_grad = loss_function.get_loss_grad()
            grad_without_reg = model.get_sum_grad_weights(loss_grad=loss_grad)

            # Find dF/dw = 1/n * sum(dE/dy * dy/dw) + dR/dw,
            # where dR/dw calculates in regularization.get_weights_reg_grad(),
            # and sum(dE/dy * dy/dw) is already calculated:
            grad_weights = (
                (1 / self.minibatch_size) * grad_without_reg +
                regularization.get_weights_reg_grad())

            # Same for bias. Find sum(dE/dy * dy/db),
            # where dE/dy - loss_grad parameter, was already calculated,
            # dy/db calculates in model.get_sum_grad_bias:
            grad_without_reg_bias = model.get_sum_grad_bias(
                loss_grad=loss_grad)

            # Find dF/db = 1/n * sum(dE/dy * dy/db) + dR/db,
            # where dR/db calculates in regularization.get_bias_reg_grad(),
            # and sum(dE/dy * dy/db) is already calculated:
            grad_bias = (
                (-1 / self.minibatch_size) * grad_without_reg_bias +
                regularization.get_bias_reg_grad())

            # Update weights and bias:
            self.weights -= grad_weights * self.learning_rate
            self.bias -= grad_bias * self.learning_rate

            # Calculate accuracy and loss:
            good_train_samples, loss = self.get_n_good_samples_from_minibatch(
                "train", self.minibatch, good_train_samples)
            train_loss.append(loss)
        self.current_loss_train = numpy.mean(train_loss)
        train_time_end = time.time()
        return good_train_samples, train_time_end - train_time_start

    def validation_epoch(self):
        """
        Validates validation dataset one time
        """
        valid_time_start = time.time()
        good_valid_samples = 0
        valid_loss = []
        if self.current_loss_valid is not None:
            self.prev_loss_valid = self.current_loss_valid
        for minibatch_index in range(self.valid_batches_n):
            self.minibatch, self.target_valid_minibatch = \
                self.get_minibatches(minibatch_index, "valid")
            # Calculate accuracy and loss:
            good_valid_samples, loss = self.get_n_good_samples_from_minibatch(
                "valid", self.minibatch, good_valid_samples)
            valid_loss.append(loss)
        self.current_loss_valid = numpy.mean(valid_loss)
        valid_time_end = time.time()
        return good_valid_samples, valid_time_end - valid_time_start

    def get_n_good_samples_from_minibatch(
            self, set_name, minibatch, good_samples):
        """
        Calculates number of good samples and loss for one
        (train or validation) minibatch
        """
        target_minibatch = getattr(self, "target_%s_minibatch" % set_name)
        prediction_minibatch = numpy.zeros_like(target_minibatch)
        for sample_index in range(minibatch.shape[0]):
            prediction = numpy.dot(
                minibatch[sample_index],
                self.weights.reshape(self.n_features)) - self.bias
            target = target_minibatch[sample_index]
            prediction_minibatch[sample_index] = prediction

            if prediction < 0:
                prediction = -1
            else:
                prediction = 1

            if prediction == target:
                good_samples += 1

        loss_function = self.loss_function(
            target=target_minibatch, prediction=prediction_minibatch)
        loss = loss_function.get_loss()

        return good_samples, loss

    def evaluation(self, good_samples, set_time, set_name):
        """
        Print some statistics (accuracy, loss) for each epoch. Saved statistics
        """
        n_samples = getattr(self, "n_%s_samples" % set_name)
        current_accuracy = (good_samples / n_samples * 100)
        current_loss = getattr(self, "current_loss_%s" % set_name)
        if set_name == "valid":
            if current_accuracy > self.max_valid_accuracy:
                self.max_valid_accuracy = current_accuracy
                print("Validation accuracy is better!!!")
                self.best_weights = copy.deepcopy(self.weights)
                self.best_bias = copy.deepcopy(self.bias)
                self.failed_iterations = 0
            else:
                self.failed_iterations += 1
        print(
            "Epoch # %s in %.2f sec. %s set: %s good from %s (%.2f " %
            (self.epoch, set_time, set_name, good_samples, n_samples,
             current_accuracy) + "%). " + "Loss: %s" % current_loss)
        self.accuracy_per_epoch[set_name].append(current_accuracy)
        self.loss_per_epoch[set_name].append(current_loss)

    def set_condition(self):
        """
        Define condition to stop training process
        """
        if self.stop_condition == "accuracy":
            condition = self.failed_iterations <= self.max_failed_iterations
        elif self.stop_condition == "loss":
            condition = (
                self.current_loss_valid is None or
                self.prev_loss_valid > self.current_loss_valid)
        else:
            NotImplementedError(
                "Please select stop_condition of 'accuracy' and 'loss'")
        return condition

    def run(self):
        """
        Trains and validates dataset until the stop condition is not met or
        number of epochs becomes equal to the maximum
        """
        condition = self.set_condition()
        while self.epoch < self.max_epochs and condition:
            good_train_samples, train_time = self.train_epoch()
            self.evaluation(good_train_samples, train_time, "train")
            good_valid_samples, valid_time = self.validation_epoch()
            self.evaluation(good_valid_samples, valid_time, "valid")
            self.epoch += 1
            condition = self.set_condition()
