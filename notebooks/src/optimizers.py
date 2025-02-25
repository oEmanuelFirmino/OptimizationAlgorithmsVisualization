import numpy as np


class Optimizer:
    def __init__(
        self,
        lr=0.01,
        l2_lambda=0.01,
        momentum=0.9,
        decay_rate=0.96,
        decay_steps=1000,
        warmup_steps=1000,
        max_lr=0.1,
    ):
        self.lr = lr
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.global_step = 0
        self.velocity = 0
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, weights, gradients):
        raise NotImplementedError

    def get_lr(self):
        self.global_step += 1
        if self.global_step < self.warmup_steps:
            return self.lr * (self.global_step / self.warmup_steps)
        return min(self.lr, self.max_lr) * np.power(
            self.decay_rate, self.global_step / self.decay_steps
        )


class SGD(Optimizer):
    def __init__(
        self,
        lr=0.01,
        l2_lambda=0.01,
        momentum=0.9,
        decay_rate=0.96,
        decay_steps=1000,
        warmup_steps=1000,
        max_lr=0.1,
    ):
        super().__init__(
            lr, l2_lambda, momentum, decay_rate, decay_steps, warmup_steps, max_lr
        )

    def update(self, weights, gradients):
        gradients += self.l2_lambda * weights
        self.velocity = self.momentum * self.velocity + gradients
        learning_rate = self.get_lr()
        return weights - learning_rate * self.velocity


class Adam(Optimizer):
    def __init__(
        self,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        l2_lambda=0.01,
        momentum=0.9,
        decay_rate=0.96,
        decay_steps=1000,
        warmup_steps=1000,
        max_lr=0.1,
    ):
        super().__init__(
            lr, l2_lambda, momentum, decay_rate, decay_steps, warmup_steps, max_lr
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, weights, gradients):
        gradients += self.l2_lambda * weights
        self.velocity = self.momentum * self.velocity + gradients
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        learning_rate = self.get_lr()
        return weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
