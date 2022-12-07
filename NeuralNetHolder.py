import ast
import pickle
from NN import NeuralNetwork

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        # Input Row is String A,B
        
        trained_model = open("E:/Masters/NN_and_Deep_Learning/Individual_Project/Assignment Code/Model2.txt",'r')
        for row in trained_model:
            trained_model = ast.literal_eval(row)
        
        filehandler = open('x_scaler.pkl','rb') 
        x_scaler = pickle.load(filehandler)
        
        filehandler = open('y_scaler.pkl','rb')
        y_scaler = pickle.load(filehandler)

        input_row_scaled = x_scaler.transform([[float(x) for x in input_row.split(",")]])
        
        if input_row_scaled[0][0] > 0:
            input_row_scaled[0][0] = -input_row_scaled[0][0]
            simple_nn = NeuralNetwork()
            output_row = simple_nn.predict(input_row_scaled[0],trained_model)
            output_row[0] = -output_row[0]
            output_row_scaled = y_scaler.inverse_transform([output_row])
        else:
            simple_nn = NeuralNetwork()
            output_row = simple_nn.predict(input_row_scaled[0],trained_model)
            output_row_scaled = y_scaler.inverse_transform([output_row])
        return output_row_scaled[0]
    
    
    
    
# model_read = open("M:/NN_and_Deep_Learning/Individual_Project/Assignment Code/Model.txt",'r')
# for row in model_read:
#     model_read = ast.literal_eval(row)

# x_scaler = joblib.load('xscaler.save') 
# y_scaler = joblib.load('yscaler.save') 

# a = np.array(input_row.split(',')).astype(np.float32)
# print(a[0])

# #print(np.array(input_row.split(',')).astype(float))
# input_row_scaled = x_scaler.transform(a.reshape(1,-1))
# print(len(input_row_scaled), input_row_scaled, type(input_row_scaled))
# outputs = forward_propagate(model_read,input_row_scaled[0])
# output_row_scaled = y_scaler.inverse_transform(np.array(outputs).reshape(1, -1))
# print(output_row_scaled)
        
        
        
        
