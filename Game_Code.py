from random import seed
import pickle
from Data_Scaling import MinMaxScaler
from NN import NeuralNetwork
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
# Data Scaling between -1 and 1
scaler_x = MinMaxScaler(-1,1)
scaler_y = MinMaxScaler(-1,1)

x = list(zip(*d_set))
x = x[:2]
x = [list(x) for x in zip(*x)]

y = list(zip(*d_set))
y = y[2:]
y = [list(y) for y in zip(*y)]

scaler_x.fit(x)
x_scaled = scaler_x.transform(x)

scaler_y.fit(y)
y_scaled = scaler_y.transform(y)

with open('x_scaler.pkl','wb') as outp_x:
    pickle.dump(scaler_x,outp_x,pickle.HIGHEST_PROTOCOL)
    
with open('y_scaler.pkl','wb') as outp_y:
    pickle.dump(scaler_y,outp_y,pickle.HIGHEST_PROTOCOL)
    
scaled_df = [x+y for x,y in zip(x_scaled,y_scaled)]

#%%
# Neural Network
simple_nn = NeuralNetwork()
simple_nn.initialize_network(2,4,2)
simple_nn.train(4,0.5,0,scaled_df)

        
    
    