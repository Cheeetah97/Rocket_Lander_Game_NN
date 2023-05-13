from random import random
from random import seed
import math
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

seed(1)

#%%
# Read DataSet
data = open("M:/NN_and_Deep_Learning/Individual_Project/Assignment Code/Game_Data.csv",'r')

d_set = []
for row in data:
    row = row.split(',')
    row = [float(value.replace('\n','')) for value in row]
    d_set.append(list(row))
    
#%%
# Min Max Scaling between -1 and 1
# def min_max_scaler_fit(d_set,min_range,max_range):
#     d_set_t = list(zip(*d_set))
#     n_scaled = []
#     for value in n_input:
#         n_std = (value - min(n_input)) / (max(n_input) - min(n_input))
#         n_scaled.append(n_std * (max_range-(min_range)) + (min_range))
#     return n_scaled

# def min_max_scaler_transform(n_input,min_range,max_range):
#     n_scaled = []
#     for value in n_input:
#         n_std = (value - min(n_input)) / (max(n_input) - min(n_input))
#         n_scaled.append(n_std * (max_range-(min_range)) + (min_range))
#     return n_scaled

# def min_max_scaler_inverse_transform(n_input,min_value,max_value,min_range,max_range):
#     n_orig = []
#     for value in n_input:
#         n_std = (value - min_range)/(max_range - min_range)
#         n_orig.append( n_std*(max_value-min_value) + min_value)
#     return n_orig

#%%
# Initialize a network
def initialize_network(n_inputs,n_hidden,n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

#%%
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    # print(len(weights),len(inputs),inputs)
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

#%%
# Transfer neuron activation
def activation_func(activation):
	return math.tanh(activation)

#%%
# Forward propagate input to a network output
def forward_propagate(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output'] = activation_func(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

#%%
# Transfer Derivative
def activation_derivative(output):
	return (1.0 + output) * (1.0 - output)

#%%
# Error Back Propagation
def backward_propagate_error(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1: # delta for hidden Layers
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:                   # delta for output Layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * activation_derivative(neuron['output'])
    return None

#%%

# Update network weights with error
def update_weights(network,row,l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']
    return None

#%%

# Scaling
# d_set_t = list(zip(*d_set))

# d_set_scaled = []
# for i in range(len(d_set_t)):
#     d_set_scaled.append(min_max_scaler_fit(d_set_t[i],-1,1))

# d_set_scaled = list(zip(*d_set_scaled))

x_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaler = MinMaxScaler(feature_range=(-1,1))

x_set_scaled = x_scaler.fit_transform(np.array(d_set)[:,:2])
y_set_scaled = y_scaler.fit_transform(np.array(d_set)[:,2:])

d_set_scaled = np.concatenate((x_set_scaled,y_set_scaled),axis=1)
d_set_scaled = d_set_scaled.tolist()

#joblib.dump(x_scaler,'xscaler.save') 
#joblib.dump(y_scaler,'yscaler.save') 

# Train a network for a fixed number of epochs
l_rate = 0.5
n_epoch = 4
network = initialize_network(2,4,2)

for epoch in range(n_epoch):
    sum_error = 0
    for row in d_set_scaled:
        expected = row[-2:]
        row = row[:-2]
        outputs = forward_propagate(network,row)
        sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        backward_propagate_error(network,expected)
        update_weights(network,row,l_rate)
    #l_rate = l_rate*0.90
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
 
#%%
# Save Trained Model

#with open('M:/NN_and_Deep_Learning/Lab-2/Assignment Code/Model.txt','w') as fp:
#    fp.write(str(network))
#    print('Done')



