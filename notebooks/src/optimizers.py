import numpy as np


class Optimizer:
    def __init__(self, l2_lambda=0.01):
        self.l2_lambda = l2_lambda  # Novo parâmetro para regularização L2

    def update(self, weights, gradients):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.01, l2_lambda=0.01):
        super().__init__(l2_lambda)
        self.lr = lr

    def update(self, weights, gradients):
        regularized_gradients = gradients + self.l2_lambda * weights
        return weights - self.lr * regularized_gradients


class Adam(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, l2_lambda=0.01):
        super().__init__(l2_lambda)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, weights, gradients):
        regularized_gradients = gradients + self.l2_lambda * weights
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * regularized_gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (regularized_gradients**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
