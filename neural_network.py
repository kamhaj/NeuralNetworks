'''
Neural network...
same as perceptron.py, but in class and with a new data to be calculated
using already trained model

'''

import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        # syn weights (from -1 to 1, mean 0)
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iter in range(training_iterations):
            outputs = self.think(training_inputs)   # normalize outputs
            error = training_outputs - outputs
            adjustment = np.dot(training_inputs.T, error * self.sigmoid_derivative(outputs))
            self.synaptic_weights += adjustment

    # used for training and for new input data (synaptic weights well-adjusted after training)
    def think(self, inputs):
        inputs = inputs.astype(float)       # synaptic weights are floats
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


def main():
    neural_network = NeuralNetwork()

    print("Random synaptic weights:\n" + str(neural_network.synaptic_weights))

    # input dataset - whenever 1 is on the first position, the output value is 1, if its 0, output is 0
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    # output dataset
    training_outputs = np.array([[0,1,1,0]]).T     #transpose to make 4x1 matrix

    neural_network.train(training_inputs, training_outputs, training_iterations=10000)

    print("Synaptic weights after training:\n" + str(neural_network.synaptic_weights))

    # take new input values - from user
    # try '0 1 0' or '0 0 0' - output near 0 - doesn't work so well (not to many examples?)
    # try '1 1 0' or '1 0 0' - output near 1
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation input data = ", A, B, C)
    print("Output data: \n")
    print(neural_network.think(inputs=np.array([A,B,C])))



if __name__ == "__main__":
    main()