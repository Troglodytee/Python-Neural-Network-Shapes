import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = [Layer(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def forward(self, x):
        self.n = x.shape[0]
        self.layers[0].values = x
        for i in range(1, len(self.layers)):
            self.layers[i].activation(self.layers[i-1].forward())
        self.outputs = softmax(self.layers[-1].forward())

    def backpropagation(self, y, learning_rate):
        self.cost = (1/self.n)*np.sum(-y*np.log(self.outputs))
        delta = (1/self.n)*(self.outputs-y)
        for i in range(len(self.layers)-1, -1, -1):
            d_theta = np.dot(self.layers[i].values.T, delta)
            d_biases = np.dot(np.ones(self.n), delta)
            d_layer = np.dot(delta, self.layers[i].weights.T)
            delta = np.multiply(d_layer, ActivationSigmoid.backpropagation(1-self.layers[i].values))
            self.layers[i].weights -= learning_rate*d_theta
            self.layers[i].biases -= learning_rate*d_biases

    def train(self, x, y, learning_rate):
        self.forward(x)
        self.backpropagation(y, learning_rate)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.random.rand(1, n_neurons)
        self.values = None

    def forward(self):
        return self.biases+np.dot(self.values, self.weights)

    def activation(self, inputs):
        self.values = ActivationSigmoid.forward(inputs)


class ActivationSigmoid:
    def forward(x):
        return 1/(1+np.exp(-x))

    def backpropagation(x):
        return x*(1-x)

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z), axis=1, keepdims=True)
