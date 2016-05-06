# -*- coding: utf-8 -*-
"""
Created on April 8, 2016
"""


class MappedObjectsRegistry(type):
    """
    General purpose mapping between names and real classes.
    Derive your registry class from this base and the attribute defined by
    "mapping" will point to the dictionary between strings and classes, which
    metaclass is your registry class.
    """
    mapping = "You must define \"mapping\" static attribute in your metaclass"
    base = object

    def __init__(cls, name, bases, clsdict):
        yours = set(cls.mro())
        mine = set(cls.base.mro())
        left = yours - mine
        mapping = getattr(type(cls), cls.mapping, {})
        if len(left) > 1 and "MAPPING" in clsdict:
            mapping[clsdict["MAPPING"]] = cls
        setattr(type(cls), cls.mapping, mapping)
        super(MappedObjectsRegistry, cls).__init__(name, bases, clsdict)
