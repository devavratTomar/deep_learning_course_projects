# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:59:50 2019

@author: devav
"""

class BaseModule(object):
    """
    This is the base class for all submodules.
    """
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    def param(self):
        pass