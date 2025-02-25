import numpy as np


def quadratic_loss(w):
    return w[0] ** 2 + 2 * w[1] ** 2


def rastrigin_loss(w):
    return 10 * len(w) + sum(w_i**2 - 10 * np.cos(2 * np.pi * w_i) for w_i in w)


def rosenbrock_loss(w):
    return 100 * (w[1] - w[0] ** 2) ** 2 + (1 - w[0]) ** 2


def hinge_loss(w, X, y, lambda_reg=0.01):
    regularization_loss = 0.5 * lambda_reg * np.sum(w**2)
    hinge_loss = np.mean(np.maximum(0, 1 - y * np.dot(X, w)))
    return hinge_loss + regularization_loss


def cross_entropy_loss(w, X, y):
    z = np.dot(X, w)
    loss = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))
    return loss


def quadratic_loss_gradient(w):
    return np.array([2 * w[0], 4 * w[1]])


def rastrigin_loss_gradient(w):
    return np.array(
        [
            2 * w[0] + 20 * np.pi * np.sin(2 * np.pi * w[0]),
            4 * w[1] + 20 * np.pi * np.sin(2 * np.pi * w[1]),
        ]
    )


def rosenbrock_loss_gradient(w):
    grad_0 = -400 * w[0] * (w[1] - w[0] ** 2) - 2 * (1 - w[0])
    grad_1 = 200 * (w[1] - w[0] ** 2)
    return np.array([grad_0, grad_1])


def hinge_loss_gradient(w, X, y, lambda_reg=0.01):
    n_samples = len(y)
    distances = 1 - y * np.dot(X, w)
    dw = np.zeros(w.shape)
    for ind, d in enumerate(distances):
        if max(0, d) == 0:
            di = w
        else:
            di = -y[ind] * X[ind, :]
        dw += di

    dw = dw / n_samples + lambda_reg * w
    return dw


def cross_entropy_loss_gradient(w, X, y):
    z = np.dot(X, w)
    sigmoid = 1 / (1 + np.exp(-z))
    grad = np.dot(X.T, (sigmoid - y)) / len(y)
    return grad
