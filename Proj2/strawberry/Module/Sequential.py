from .base_module import BaseModule

class Sequential(BaseModule):
    """
    Implements MLP with layers Linear, RelU, Tanh.
    """
    def __init__(self, *layers):
        #TODO: check if inputs are of type BaseModule
        self.layers = [layer for layer in layers]
    
    def forward(self, inputs):
        """
        Performs foward pass through all layers.
        """
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, gradwrtoutput):
        """
        Performs backpropagation through all layers
        """
        grad_out = gradwrtoutput
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)

        return grad_out

    def param(self):
        params = []
        for layer in self.layers:
            for param in layer.param():
                params.append(param)
        return params