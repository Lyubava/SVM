# -*-coding: utf-8 -*-
"""
Created on April 8, 2016
"""


from six import add_metaclass
from zope.interface import implementer, Interface

from mapped_object_registry import MappedObjectsRegistry


class IRegularization(Interface):
    def get_weights_reg_grad():
        """
        Calculates dR/dw, where R(w, b) - regularization
        """

    def get_bias_reg_grad():
        """
        Calculates dR/db, where R(w, b) - regularization
        """
        
        
class RegularizationRegistry(MappedObjectsRegistry):
    mapping = "regularization"


@add_metaclass(RegularizationRegistry)
class BaseRegularization(object):
    def __init__(self, lambda_coeff, **kwargs):
        self.lambda_coeff = lambda_coeff
         
         
@implementer(IRegularization)
class L1Regularization(BaseRegularization):
    MAPPING = "l1"

    def get_weights_reg_grad(self):
        return self.lambda_coeff

    @staticmethod
    def get_bias_reg_grad():
        return 0


@implementer(IRegularization)
class L2Regularization(BaseRegularization):
    MAPPING = "l2"

    def __init__(self, **kwargs):
        super(L2Regularization, self).__init__(**kwargs)
        self.weights = kwargs["weights"]

    def get_weights_reg_grad(self):
        return 2 * self.lambda_coeff * self.weights

    @staticmethod
    def get_bias_reg_grad():
        return 0
