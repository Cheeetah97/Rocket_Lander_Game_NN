from random import seed
import pickle
from Data_Scaling import MinMaxScaler
from NN import NeuralNetwork
seed(1)

#%%
# Read DataSet
train_data = open("F:/Masters/NN_and_Deep_Learning/Individual_Project/Assignment/TRAIN_DATA.csv",'r')
valid_data = open("F:/Masters/NN_and_Deep_Learning/Individual_Project/Assignment/TEST_DATA.csv",'r')

# Bringing the Data Sets into List of List format
train_d_set = []
for row in train_data:
    row = row.split(',')
    row = [float(value.replace('\n','')) for value in row]
    train_d_set.append(list(row))
    
valid_d_set = []
for row in valid_data:
    row = row.split(',')
    row = [float(value.replace('\n','')) for value in row]
    valid_d_set.append(list(row))
    
#%%
# Data Scaling between -1 and 1
# Using the MinMax scaler module created by me
scaler_x = MinMaxScaler(0,1)
scaler_y = MinMaxScaler(0,1)

x = list(zip(*train_d_set))
x = x[:2]
x = [list(x) for x in zip(*x)]

y = list(zip(*train_d_set))
y = y[2:]
y = [list(y) for y in zip(*y)]

x_valid = list(zip(*valid_d_set))
x_valid = x_valid[:2]
x_valid = [list(x_valid) for x_valid in zip(*x_valid)]

y_valid = list(zip(*valid_d_set))
y_valid = y_valid[2:]
y_valid = [list(y_valid) for y_valid in zip(*y_valid)]

scaler_x.fit(x)
x_scaled = scaler_x.transform(x)
x_valid_scaled = scaler_x.transform(x_valid)

scaler_y.fit(y)
y_scaled = scaler_y.transform(y)
y_valid_scaled = scaler_y.transform(y_valid)

# Saving the Scaler Objects
with open('x_scaler.pkl','wb') as outp_x:
    pickle.dump(scaler_x,outp_x,pickle.HIGHEST_PROTOCOL)
    
with open('y_scaler.pkl','wb') as outp_y:
    pickle.dump(scaler_y,outp_y,pickle.HIGHEST_PROTOCOL)
    
scaled_df = [x+y for x,y in zip(x_scaled,y_scaled)]
scaled_valid_df = [x+y for x,y in zip(x_valid_scaled,y_valid_scaled)]

#%%
# Neural Network
ann = NeuralNetwork()
ann.network_intialization(2,8,2)
ann.train(500,0.1,0,train_set=scaled_df,eval_set=scaled_valid_df,early_stopping_rounds=50)


        
    
    