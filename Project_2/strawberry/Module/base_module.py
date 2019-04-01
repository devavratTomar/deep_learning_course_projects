class BaseModule(object):
    """
    This is the base class for all submodules.
    """    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
    def param(self):
        raise NotImplementedError