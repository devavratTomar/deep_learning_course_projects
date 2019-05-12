from .base_module import BaseModule
import torch
from strawberry.utils import get_accuracy

class Sequential(BaseModule):
    """
    Implements MLP with layers Linear, RelU, Tanh.
    """
    def __init__(self, *layers):
        #TODO: check if inputs are of type BaseModule
        self.layers = [layer for layer in layers]
        self.history = dict()
        self.history['loss']=[]
        self.history['val_loss']=[]
        self.history['acc']=[]
        self.history['val_acc']=[]
    
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
        Performs backward pass through this module.
        
        :param gradwrtoutput: gradient of loss with respect to the output of this module. shape should be [batch_size, out_features]
        
        :return: gradient of loss with respect to the input of this module.
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
    
    
    def _get_moving_average(self, prediction, train_labels, prev_avg, nb, metric):
        """
        Calculates moving average for the given metric and prediction
        
        :param train_data:    training data
        :param train_labels:  labels of the training data
        :param prev_avg:      previous average of the metric
        :param nb:            number of values averaged
        :param metric:        metric 
        
        :return:              moving average
        """
                        
        # Forward predictions to metric
        value = metric(prediction, train_labels)
                
        # Moving average 
        new_avg = (value + max(nb, 1) * prev_avg) / (nb+1)
        
        return new_avg
                

    
    def train(self, x_train, y_train, epochs, batch_size, opt, loss, accuracy=False, 
              validation_set=None, verbose=False):
        """
        Trains the model on the given data with specified parameters
            
        :param x_train:          training data 
        :param y_train:          labels of the training data
        :param epochs:           epochs for training
        :param batch_size:       batch size
        :param opt:              optimizer
        :param loss:             loss      
        :param accuracy:         whether to monitor the accuracy
        :param validation_set:   (validation data, labels)
        :param verbose:          whether to print the progress
        
        :return: gradient of loss with respect to the input of this module. 
        """
        n_batches = x_train.shape[0]//batch_size
        
        for epoch in range(epochs):
            
            avg_loss = 0.0
            
            if accuracy:
                avg_accuracy = 0.0
            
            for batch in range(n_batches):
                
                train_data   = x_train[batch*batch_size:(batch+1)*batch_size, :]
                train_labels = y_train[batch*batch_size:(batch+1)*batch_size, :]
                
                # Update loss average
                model_output = self.forward(train_data)
                avg_loss = self._get_moving_average(model_output, train_labels, prev_avg=avg_loss, nb=batch, 
                                                    metric=loss)
                
                if accuracy:
                    # Update the accuracy average
                    pred = self.predict_from_output(model_output)
                    avg_accuracy = self._get_moving_average(pred, train_labels, prev_avg=avg_accuracy, nb=batch, 
                                                            metric=get_accuracy)
                
                # Backward of loss so that gradients are accumulated in it. Then backword of Model to accumulate its gradients.
                self.backward(loss.backward())
        
                opt.step(self.param())
                
               
            # Update loss average
            model_output = self.forward(train_data)
            avg_loss = self._get_moving_average(model_output, train_labels, prev_avg=avg_loss,
                                               nb=epochs, metric=loss)
            self.history['loss'].append(avg_loss)
                
            if accuracy:
                    # Update the accuracy average
                    pred = self.predict_from_output(model_output)
                    avg_accuracy = self._get_moving_average(pred, train_labels, prev_avg=avg_accuracy,
                                               nb=batch, metric=get_accuracy) 
                    self.history['acc'].append(avg_accuracy)
                    
            if validation_set is not None:
                # Predict on validation data
                val_pred = self.forward(validation_set[0])
                val_loss = loss(val_pred, validation_set[1])                                   
                self.history['val_loss'].append(val_loss)
                
                if accuracy:
                    val_acc = get_accuracy(val_pred, validation_set[1])
                    self.history['val_acc'].append(val_acc)
        
            if verbose:
                info_msg = "Epoch: {}, Training loss: {:.4f}".format(epoch, self.history['loss'][-1])
                
                if accuracy:
                    info_msg += ", Acc.: {0:.2f}%".format(self.history['acc'][-1])
                    
                if validation_set is not None:
                    info_msg += ", Validation Loss: {0:.4f}".format(self.history['val_loss'][-1])
                    if accuracy:
                        info_msg += ", Validation Acc.: {0:.2f}%".format(self.history['val_acc'][-1])
                        
                print(info_msg)
            
        
    def predict(self, x):  
        """
        Predicst labels and returns one hot vector
        
        :param x_test:     input to predict its labels  
            
        :return:           prediction 
        """
        
        model_output = self.forward(x)
        return self.predict_from_output(model_output)


    def predict_from_output(self, model_output):
        """
        Predicst labels and returns one hot vector
        
        :param x_test:     output of the model  
            
        :return:           prediction 
        """
        
        # Create one hot encoding of predicted labels
        max_idx = torch.argmax(model_output, 1, keepdim=True)
        one_hot = torch.FloatTensor(model_output.shape)
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        
        return one_hot
        
        