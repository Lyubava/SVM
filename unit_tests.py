#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
Created on April 8, 2016
"""


import numpy
import matplotlib.pyplot as plt
import unittest

from sgd import SGD


class SVMUnitTest(unittest.TestCase):
    def test_svm_points(self):
        test_input_data = numpy.array(
            [[2, 1], [-2, 1], [2, 1], [-2, 1]], numpy.float32)
        test_target_data = numpy.array([-1, 1, -1, 1], numpy.float32)
        test_target_data = test_target_data.reshape(
            test_target_data.shape[0], 1)
        sgd = SGD(
            test_input_data, test_target_data, loss_type="hinge",
            regularization_type="l2", learning_rate=0.6, lambda_coeff=0.1,
            max_epochs=100, max_failed_iterations=10, minibatch_size=1,
            n_valid_samples=2, initialization="random",
            stop_condition="accuracy", model="linear_svm")
        sgd.run()

        self.visualize_dots(
            test_input_data, test_target_data, sgd.best_weights, sgd.best_bias)

        assert sgd.max_valid_accuracy == 100.0

    def test_svm_dotes(self):
        test_input_data = numpy.array(
            [[1, -4], [2, -2], [5, -3], [2, 3], [1, 4], [3, 5], [1, -4],
             [2, -2], [5, -3], [2, 3], [1, 4], [3, 5]], numpy.float32)
        test_target_data = numpy.array(
            [-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1], numpy.float32)

        test_target_data = test_target_data.reshape(
            test_target_data.shape[0], 1)

        sgd = SGD(
            test_input_data, test_target_data, loss_type="hinge",
            regularization_type="l2", learning_rate=0.1, lambda_coeff=0.5,
            max_epochs=100, max_failed_iterations=10, minibatch_size=1,
            n_valid_samples=6)
        sgd.run()

        self.visualize_dots(
            test_input_data, test_target_data, sgd.best_weights, sgd.best_bias)

        assert sgd.max_valid_accuracy == 100.0

    def visualize_dots(
            self, test_input_data, test_target_data, best_weights, best_bias):
        neg_x_points, neg_y_points, pos_x_points, pos_y_points = \
            self.get_x_y_points(test_input_data, test_target_data)

        self.visualize_points(
            neg_x_points, neg_y_points, pos_x_points, pos_y_points)

        max_value = self.get_max_abs_point(
            neg_x_points + pos_x_points, neg_y_points + pos_y_points)

        self.visualize_line_from_weights_bias(
            best_weights, best_bias, max_value)

        plt.show()

    def get_x_y_points(self, input_data, target_data):
        neg_x_points = []
        neg_y_points = []
        pos_x_points = []
        pos_y_points = []
        for point_index in range(input_data.shape[0]):
            point = input_data[point_index]
            if target_data[point_index] > 0:
                pos_x_points.append(point[0])
                pos_y_points.append(point[1])
            else:
                neg_x_points.append(point[0])
                neg_y_points.append(point[1])

        return neg_x_points, neg_y_points, pos_x_points, pos_y_points

    def get_max_abs_point(self, x_points, y_points):
        max_x = max(min(x_points), max(x_points), key=abs)
        max_y = max(min(y_points), max(y_points), key=abs)
        return max(max_x, max_y)

    def visualize_line_from_weights_bias(
            self, best_weights, best_bias, max_value):
        x_zero = best_weights[0][0]
        y_zero = best_weights[1][0]
        if x_zero != 0.0 and y_zero != 0.0:
            coeff = x_zero / y_zero
            if best_bias != 0.0:
                y_one = - (
                    best_bias / (numpy.sqrt(1 + coeff ** 2))) + y_zero
                x_one = y_one * coeff
                x_two, y_two = y_one, -x_one
                alpha = (y_two - y_one) / (x_two - x_one)
                beta = y_two - alpha * x_two
                x_three = x_two + 2 * max_value
                y_three = alpha * x_three + beta
            else:
                x_one = x_zero
                y_one = y_zero
                x_two, y_two = (x_one + ((y_one ** 2) / x_one)), 0
                alpha = (y_two - y_one) / (x_two - x_one)
                beta = y_two - alpha * x_two
                x_three = x_two + 2 * max_value
                y_three = alpha * x_three + beta
        elif x_zero == 0.0:
            y_one = y_zero - best_bias
            x_one = max_value
            x_three = - max_value
            y_three = y_one
        else:
            x_one = x_zero - best_bias
            x_three = x_one
            y_three = - max_value
            y_one = max_value
        plt.plot([x_one, x_three], [y_one, y_three])

    def visualize_points(
            self, neg_x_points, neg_y_points, pos_x_points, pos_y_points):
        plt.plot(pos_x_points, pos_y_points, 'ro')
        plt.plot(neg_x_points, neg_y_points, 'bo')


if __name__ == "__main__":
    unittest.main()
