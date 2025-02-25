import numpy as np


def quadratic_loss(w):
    return w[0] ** 2 + 2 * w[1] ** 2


def rastrigin_loss(w):
    return 10 * len(w) + sum(w_i**2 - 10 * np.cos(2 * np.pi * w_i) for w_i in w)


def rosenbrock_loss(w):
    return 100 * (w[1] - w[0] ** 2) ** 2 + (1 - w[0]) ** 2


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
