import numpy as np


class SimpleNeuralNet:
    def __init__(self):
        self.weights1 = np.random.randn(2, 3)
        self.weights2 = np.random.randn(3, 1)

    def forward(self, X):
        hidden = np.maximum(0, np.dot(X, self.weights1))
        output = 1 / (1 + np.exp(-np.dot(hidden, self.weights2)))
        return hidden, output

    def backward(self, X, y, hidden, output):
        loss = np.mean((output - y) ** 2)

        d_output = 2 * (output - y) * output * (1 - output)
        d_hidden = np.dot(d_output, self.weights2.T) * (hidden > 0)

        grad_weights2 = np.dot(hidden.T, d_output)
        grad_weights1 = np.dot(X.T, d_hidden)

        return loss, grad_weights1, grad_weights2
