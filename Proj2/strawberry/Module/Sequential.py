from .base_module import BaseModule

class Sequential(BaseModule):
    """
    Implements sequential neural network
    """
    
    def __init__(self, *args):
        for module in args:
            self.add_module(module)
    
    def forward(self, inputs):
        for module in self.child_modules:
            inputs = module.forward(inputs)
        return inputs
        
    
    def add_module(self, module):
        if module is None:
            raise TypeError("Module can not be None")
        elif not isinstance(module, BaseModule):
            raise TypeError("Module should be a child of the BaseModule")
            
        # array of ordered child modules
        self.child_modules.append(module)