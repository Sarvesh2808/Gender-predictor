import numpy as np


def sigmoid(x):
    # Activation function: 1/(1 + exp^-x)
    return 1/(1 + np.exp(-x))


class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias and then activation function sigmoid
        total = np.dot(self.weight, inputs) + self.bias
        return sigmoid(total)


weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
print(n.feedforward(x))


#  Now building neural network
class MyNeuralNetwork:

    '''
    A neural network with
        - 2 inputs
        - 2 hidden layer neurons (h1, h2)
        - an output layer with one
    Each neuron have same weight and bias
        weight = [0, 1]
        bias = 0
    '''

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # The neuron class
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The input to o1 is output from h1 & h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = MyNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))