# -*-coding: utf-8 -*-
"""
Created on April 8, 2016
"""


import numpy
from six import add_metaclass
from zope.interface import implementer, Interface

from mapped_object_registry import MappedObjectsRegistry


class IModel(Interface):
    def get_sum_grad_weights(self, loss_grad):
        """
        :param loss_grad:
        :return:
        """
    def get_sum_grad_bias(self, loss_grad):
        """
        :param loss_grad:
        :return:
        """


class ModelsRegistry(MappedObjectsRegistry):
    mapping = "models"


@add_metaclass(ModelsRegistry)
class BaseModel(object):
    pass


@implementer(IModel)
class LinearSVM(BaseModel):
    MAPPING = "linear_svm"

    def __init__(self, n_features, minibatch_size, minibatch):
        self.n_features = n_features
        self.minibatch_size = minibatch_size
        self.minibatch = minibatch

    def get_sum_grad_weights(self, loss_grad):
        """
        Calculates sum(dE/dy * dy/dw), where E(y) - loss function,
        dE/dy - loss function partial derivative on y,
        y = wx - b - linear svm model,
        dy/dw - partial derivative on weights of linear svm model and
        it equals x, i.e. self.minibatch
        :param loss_grad: dE/dy, loss function partial derivative on y
        :return: sum(dE/dy * dy/dw)
        """
        grad_weights = self.minibatch
        return numpy.dot(numpy.transpose(grad_weights), loss_grad)

    def get_sum_grad_bias(self, loss_grad):
        """
        Calculates sum(dE/dy * dy/db), where E(y) - loss function,
        dE/dy - loss function partial derivative on y,
        y = wx - b - linear svm model,
        dy/db - partial derivative on bias of linear svm model and
        it equals -1
        :param loss_grad: dE/dy, loss function partial derivative on y
        :return: sum(dE/dy * dy/db)
        """
        return - sum(loss_grad)[0]

