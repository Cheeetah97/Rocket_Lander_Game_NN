from random import random,shuffle
from math import exp

class NeuralNetwork:
    
    def __init__(self):
        self.network = []
        
    # Initializing the Neural Network with 1 Input Layer, 1 Hidden Layer and 1 Output Layer
    # The function takes inputs for the number of Input,Hidden and Output Neurons
    def network_intialization(self,n_input,n_hidden,n_output):
        hidden_layer = [{'weights':[random() for i in range(n_input + 1)],
                         'c_weights':[0.0 for i in range(n_input + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)],
                         'c_weights':[0.0 for i in range(n_hidden + 1)]} for i in range(n_output)]
        self.network.append(output_layer)
        
    # Calculate Neuron Activation for an Input
    # Neuron Activation is the product of the Synaptic Weight associated with it times the Input
    def _activate(self,weights,inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    # We are using Tanh activation funtion
    def _activation_func(self,activation):
        return (exp(2*activation)-1)/(exp(2*activation)+1)
    
    # Forward Propagating the Inputs to generate Output
    # Simple for loop over the Layers and Neurons and calling the functions defined above
    def _forward_propagation(self,row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                neuron['output'] = self._activation_func(self._activate(neuron['weights'],inputs))
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    # Derivative of the Transfer Function
    # This Derivative is needed to calculate the local gradient for Back Propagation
    # The Derivative of Tanh is simply (1-Output)(1+Output)
    def _activation_derivative(self,output):
     	return ((1.0 + output) * (1.0 - output))

    # Error Back Propagation
    # We iterate over each Layer Backwards
    # Delta for the Output Layer is simply (output - expected) * transfer_derivative(output)
    # Delta for the Hidden Layer is simply (weight_k * error_j) * transfer_derivative(output)
    # where error_j is the Eerror signal from the jth Neuron in the Output layer
    def _back_propagation(self,expected):
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
                    errors.append(expected[j]-neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self._activation_derivative(neuron['output'])
        return None

    # Update Network Weights with Error
    # The Weights are updated using the formula w_new = w_old - (lr*delta*input) + (momentum*changeinweights)
    
    def _update_weights(self,row):
        for i in range(len(self.network)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += (self.l_rate * neuron['delta'] * inputs[j]) #+ (self.momentum * neuron['c_weights'][j])
                    #neuron['c_weights'][j] = (self.l_rate * neuron['delta'] * inputs[j]) + (self.momentum * neuron['c_weights'][j])
                neuron['weights'][-1] += (self.l_rate * neuron['delta']) #+ (self.momentum * neuron['c_weights'][-1])
                #neuron['c_weights'][-1] = (self.l_rate * neuron['delta']) + (self.momentum * neuron['c_weights'][-1])
        return None
    
    # Training the Neural Network
    # The User inputs the learning rate, momentum, training data set, validation data set, and early stopping rounds
    # The Model with the lowest error over the validation set in the given epochs is saved
    def train(self,epochs,l_rate,momentum,train_set,eval_set,early_stopping_rounds):
        self.l_rate = l_rate
        self.epochs = epochs
        self.momentum = momentum
        loss = {"train_loss":[],"valid_loss":[]}
        for epoch in range(epochs):
            sum_error = 0
            shuffle(train_set)
            for row in train_set:
                expected = row[-2:]
                row = row[:-2]
                #print(row,expected)
                outputs = self._forward_propagation(row)
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self._back_propagation(expected)
                self._update_weights(row)
            loss["train_loss"].append(sum_error/len(train_set))
            if epoch == 0:
                init_eval_error = 0
                initial_epoch = epoch
                for row in eval_set:
                    expected = row[-2:]
                    row = row[:-2]
                    outputs = self._forward_propagation(row)
                    init_eval_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                init_eval_error = init_eval_error/len(eval_set)
                loss["valid_loss"].append(init_eval_error)
                print(f'Validation Loss improved from inf to {init_eval_error:.3f}...Saving Model!')
                print('>epoch=%d, lrate=%.3f, train_loss=%.6f, val_loss=%.3f' % (epoch, l_rate, sum_error/len(train_set), init_eval_error))
            else:
                curr_eval_error = 0
                for row in eval_set:
                    expected = row[-2:]
                    row = row[:-2]
                    outputs = self._forward_propagation(row)
                    curr_eval_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                curr_eval_error = curr_eval_error/len(eval_set)
                loss["valid_loss"].append(curr_eval_error)
                if curr_eval_error < init_eval_error:
                    with open('F:/Masters/NN_and_Deep_Learning/Individual_Project/Assignment/Trained_Model.txt','w') as fp:
                        fp.write(str(self.network))
                        print(f'Validation Loss improved from {init_eval_error:.3f} to {curr_eval_error:.3f}...Saving Model!')
                        print('>epoch=%d, lrate=%.3f, train_loss=%.6f, val_loss=%.3f' % (epoch, l_rate, sum_error/len(train_set), curr_eval_error))
                    init_eval_error = curr_eval_error
                    initial_epoch = epoch
                elif epoch - initial_epoch > early_stopping_rounds:
                    break
                else:
                    print('>epoch=%d, lrate=%.3f, train_loss=%.6f, val_loss=%.3f' % (epoch, l_rate, sum_error/len(train_set), curr_eval_error))
                    
            with open('F:/Masters/NN_and_Deep_Learning/Individual_Project/Assignment/Trained_Model_History.txt','w') as fp:
                fp.write(str(loss))
        
            
    # Predict Output  
    # Trained Model is Used to generate predictions during gameplay
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