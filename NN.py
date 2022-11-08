from random import random
import math

class NeuralNetwork:
    
    def __init__(self):
        self.network = []
        
    # Initialize Network
    def initialize_network(self,n_input,n_hidden,n_output):
        hidden_layer = [{'weights':[random() for i in range(n_input + 1)],'c_weights':[0.0 for i in range(n_input + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)],'c_weights':[0.0 for i in range(n_hidden + 1)]} for i in range(n_output)]
        self.network.append(output_layer)
        
    # Calculate Neuron Activation for an Input
    def _activate(self,weights,inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def _activation_func(self,activation):
    	return math.tanh(activation)
    
    # Forward Propagate Input to a Network Output
    def _forward_propagate(self,row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self._activate(neuron['weights'],inputs)
                neuron['output'] = self._activation_func(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    # Transfer Derivative
    def _activation_derivative(self,output):
     	return (1.0 + output) * (1.0 - output)

    # Error Back Propagation
    def _backward_propagate_error(self,expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network)-1: # delta for hidden Layers
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:                        # delta for output Layer
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self._activation_derivative(neuron['output'])
        return None

    # Update Network Weights with Error
    def _update_weights(self,row):
        for i in range(len(self.network)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= (self.l_rate * neuron['delta'] * inputs[j]) + (self.momentum * neuron['c_weights'][j])
                    neuron['c_weights'][j] = (self.l_rate * neuron['delta'] * inputs[j]) + (self.momentum * neuron['c_weights'][j])
                neuron['weights'][-1] -= (self.l_rate * neuron['delta']) + (self.momentum * neuron['c_weights'][-1])
                neuron['c_weights'][-1] = (self.l_rate * neuron['delta']) + (self.momentum * neuron['c_weights'][-1])
        return None
    
    # Train Network
    def train(self,epochs,l_rate,momentum,d_set):
        self.l_rate = l_rate
        self.epochs = epochs
        self.momentum = momentum
        
        for epoch in range(epochs):
            sum_error = 0
            for row in d_set:
                expected = row[-2:]
                row = row[:-2]
                outputs = self._forward_propagate(row)
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self._backward_propagate_error(expected)
                self._update_weights(row)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            
        with open('M:/NN_and_Deep_Learning/Individual_Project/Assignment Code/Model2.txt','w') as fp:
            fp.write(str(self.network))
            print('############ Model Saved! ##############')
            
    # Predict Output   
    def predict(self,input_row,trained_model):
        inputs = input_row
        for layer in trained_model:
            new_inputs = []
            for neuron in layer:
                activation = self._activate(neuron['weights'],inputs)
                neuron['output'] = self._activation_func(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs