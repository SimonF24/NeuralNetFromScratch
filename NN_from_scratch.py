import numpy as np

class NeuralNet:
    def __init__(self, layers, learning_rate=1e-2):
        ''' 
        Layers is expected to be a list with the number of neurons in each layer at the appropriate index of the list
        The first layer is the number of inputs and the last layer is the number of outputs
        '''
        biases = []
        Weights = []
        for index, value in enumerate(layers):
            if index == 0:
                pass
            else:
                Weights.append(np.random.random([layers[index-1],value]))
                biases.append(np.random.random([1,value]))
        self.b = biases
        self.L = len(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        self.Weights = Weights

    def compute_mean_squared_error_loss(self, output, actual_result):
        differences = []
        for index, value in enumerate(actual_result):
            differences.append(value - output[0][index])
        return np.mean(np.array(differences)**2), differences

    def forward(self, input_data):
        '''
        Inputs:
        inputs_data: np.array
        '''
        activations = [input_data]
        layer_result = input_data
        for index in range(self.L-1):
            W = self.Weights[index]
            z = self.b[index] + np.matmul(layer_result, W)
            layer_result = z * (z > 0) # This is using a ReLU activation function
            activations.append(layer_result)
        return layer_result, activations

    def predict(self, input_data):
        return self.forward(input_data)[0]

    def train(self, input_data, actual_result):
        '''
        input_data and result are expected to be numpy arrays of the appropriate length
        '''
        output, activations = self.forward(input_data)
        loss, differences = self.compute_mean_squared_error_loss(output, actual_result)
        derived_activations = [] 
        for activation in activations: 
            # The ReLU derivative is the Heaviside (with 0 for x=0)
            # Generally, the function would be different for each layer as each layer could have a different activation function
            derived_activations.append(np.heaviside(activation, 0))
        errors = [[]] * len(self.Weights)
        temp_weights = [[]] * len(self.Weights)
        for index in range(len(self.Weights)): 
            reversed_index = -1 - index
            if index == 0:
                # The second argument to the below is the derivative of the mean squared loss function and would be different for different loss metrics
                errors[reversed_index] = np.multiply(derived_activations[reversed_index], -2 * np.array(differences)) 
                gradient_matrix = np.matmul(np.transpose(activations[reversed_index - 1]), errors[reversed_index])
                temp_weights[reversed_index] = self.Weights[reversed_index] - self.learning_rate * gradient_matrix
                self.b[reversed_index] = self.b[reversed_index] - self.learning_rate * errors[reversed_index]
            else:
                temp = np.matmul(errors[reversed_index + 1], np.transpose(self.Weights[reversed_index + 1]))
                errors[reversed_index] = np.multiply(temp, derived_activations[reversed_index])
                gradient_matrix = np.matmul(np.transpose(activations[reversed_index - 1]), errors[reversed_index])
                temp_weights[reversed_index] = self.Weights[reversed_index] - self.learning_rate * gradient_matrix
                self.b[reversed_index] = self.b[reversed_index] - self.learning_rate * errors[reversed_index]
        self.Weights = temp_weights

# The below runs through creating a training the network, showing the weights and biases at each step
net = NeuralNet(np.array([2,3,3,2]))
inputs = np.array([[1,1]])
print('Weights:\n', net.Weights)
forward_result = net.predict(inputs)
print('Biases:\n', net.b)
print('Forward Result:\n', forward_result)
actual_result = [1,1]
loss, _ = net.compute_mean_squared_error_loss(forward_result, actual_result)
print('Loss:\n', loss)
net.train(inputs, [1,1])
print('Trained Weights:\n', net.Weights)
print('Trained Biases:\n', net.b)
new_forward_result = net.predict(inputs)
new_loss, _ = net.compute_mean_squared_error_loss(new_forward_result, actual_result)
print('Forward Result After Training:\n', new_forward_result)
print('Loss After Training:\n', new_loss)
net.train(inputs, [1,1])
new_forward_result = net.predict(inputs)
new_loss, _ = net.compute_mean_squared_error_loss(new_forward_result, actual_result)
print('Final Result:\n', new_forward_result)
print('Loss After Training Again:\n', new_loss)
print('Final Weights:\n', net.Weights)
print('Final Biases:\n', net.b)