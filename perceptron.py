'''

Training process:
    1. Take the inputs from the training example and put them through our formula
       to get the neuron's output.
    2. Calculate the error, which is the difference between the output we got and
       the actual output
    3. Depending on the severeness of the error, adjust the weights accordingly.
    4. Repeat this process 20 000 times.

(adjust weight by = error.input.sigma'(output)

'''
import numpy as np

# normalizing function
def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1-x)

# input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output dataset
training_outputs = np.array([[0,1,1,0]]).T     #transpose to make 4x1 matrix

# set random numbers to make calculations
np.random.seed(1)

# initialize synaptic weights randomly with mean 0 to create weight matrix
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: \n' + str(synaptic_weights))

for iter in range(100000):
    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(x=np.dot(input_layer, synaptic_weights))

    # calculate current error
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights ater training: \n' + str(synaptic_weights))
print('Outputs after training: \n' + str(outputs))