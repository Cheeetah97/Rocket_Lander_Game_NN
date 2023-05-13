import ast
import pickle
from NN import NeuralNetwork

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        # Input Row is String A,B
        
        # Loading the Best Trained Model
        trained_model = open("F:/Masters/NN_and_Deep_Learning/Individual_Project/Assignment/Trained_Model.txt",'r')
        for row in trained_model:
            trained_model = ast.literal_eval(row)
        
        # Loading X_Scaler and Y_Scaler objects
        filehandler = open('x_scaler.pkl','rb') 
        x_scaler = pickle.load(filehandler)
        
        filehandler = open('y_scaler.pkl','rb')
        y_scaler = pickle.load(filehandler)

        # Scaling the Input Row
        input_row_scaled = x_scaler.transform([[float(x) for x in input_row.split(",")]])
        
        simple_nn = NeuralNetwork()
        output_row = simple_nn.predict(input_row_scaled[0],trained_model)
        output_row_scaled = y_scaler.inverse_transform([output_row])
        
        # Changing X with Y and Y with X
        temp = output_row_scaled[0][:]
        output_row_scaled[0][0] = temp[1]
        output_row_scaled[0][1] = temp[0]
        
        return output_row_scaled[0]