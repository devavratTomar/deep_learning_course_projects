from .base_optimizer import Optimizer
import torch

class ADAM(Optimizer):
    """
    Stochastic gradient descent with Adaptive learning rate and momentum
    
    :param lr: learning rate of stochastic gradient descent
    :param betas: coefficients to update the moving avarages
    :param eps: term to avoid numerical instability 
    """
    
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, step_cnt =0):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.step_cnt = step_cnt
        self.moving_avg_1 = []
        self.moving_avg_2 = []
    
    def step(self, params):
        """
        Computes one gradient steps of loss and applies gradients to model params.
        """
        # total number of gradient update steps
        self.step_cnt += 1
        
        beta1, beta2 = self.betas
        
        for i, param in enumerate(params):
        	if self.step_cnt == 1:
        		self.moving_avg_1.append(param[1].clone())
        		self.moving_avg_2.append(param[1].clone()**2)
        		
        	# Update biased moment estimates
        	self.moving_avg_1[i] = beta1 * self.moving_avg_1[i] + (1. - beta1) * param[1]
        	self.moving_avg_2[i] = beta2 * self.moving_avg_2[i] + (1. - beta2) * param[1]**2
            
            # Bias-corrected moment estimates
        	mhat = self.moving_avg_1[i] / (1. - beta1**self.step_cnt)
        	vhat = self.moving_avg_2[i] / (1. - beta2**self.step_cnt)
        	
        	# Update parameters    
        	param[0] -= self.lr*mhat / (torch.sqrt(vhat) + self.eps)
