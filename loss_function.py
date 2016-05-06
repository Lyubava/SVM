# -*-coding: utf-8 -*-
"""
Created on April 8, 2016
"""


import numpy
import scipy
from six import add_metaclass
from zope.interface import implementer, Interface

from mapped_object_registry import MappedObjectsRegistry


class ILossFunction(Interface):
    def get_loss_grad():
        """
        Calculates dE/dy, where E(y) - loss function, y - prediction
        """

    def get_loss():
        """
        Calculates average loss for one minibatch, given target and prediction
        """


class LossFunctionsRegistry(MappedObjectsRegistry):
    mapping = "loss_functions"


@add_metaclass(LossFunctionsRegistry)
class BaseLossFunction(object):
    epsilon = 1e-15

    def __init__(self, target, prediction, **kwargs):
        self.target = target
        self.prediction = prediction


@implementer(ILossFunction)
class LogLossFunction(BaseLossFunction):
    MAPPING = "log"

    def get_loss(self):
        prediction = scipy.maximum(self.epsilon, self.prediction)
        prediction = scipy.minimum(1 - self.epsilon, prediction)
        loss = sum(
            self.target * scipy.log(prediction) +
            scipy.subtract(1, self.target) *
            scipy.log(scipy.subtract(1, prediction)))
        loss = loss * - 1.0/self.target.shape[0]
        return loss

    def get_loss_grad(self):
        # TODO: add get_loss_grad for log loss
        pass


@implementer(ILossFunction)
class HingeLossFunction(BaseLossFunction):
    MAPPING = "hinge"

    """
    E(y) = max(0, 1 - t * y), t - target, y - prediction
    """

    def get_loss(self):
        loss = 0
        count = self.target.shape[0]
        for index in range(count):
            loss += max(0, 1 - self.target[index] * self.prediction[index])
        return loss / count

    def get_loss_grad(self):
        if numpy.dot(numpy.transpose(self.target), self.prediction)[0][0] >= 1:
            return numpy.zeros_like(self.target)
        else:
            return - self.target
