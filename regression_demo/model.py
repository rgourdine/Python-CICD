"""A tiny NumPy example: learn the line y = w * x + b with gradient descent."""

import numpy as np


def make_data(n=100, weight=2.0, bias=1.0, noise=0.0, seed=0):
    """Return (x, y) arrays of shape (n,) sampled from y = weight * x + bias + noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y = weight * x + bias + noise * rng.standard_normal(n)
    return x, y


def mse(pred, target):
    """Mean squared error."""
    return float(np.mean((pred - target) ** 2))


class LinearRegressor:
    """Fits y = weight * x + bias by gradient descent on mean squared error."""

    def __init__(self, weight=0.0, bias=0.0):
        self.weight = weight
        self.bias = bias

    def predict(self, x):
        return self.weight * np.asarray(x) + self.bias

    def fit(self, x, y, epochs=500, lr=0.1):
        """Train in place. Returns the loss for each epoch."""
        x, y = np.asarray(x), np.asarray(y)
        losses = []
        for _ in range(epochs):
            error = self.predict(x) - y
            losses.append(float(np.mean(error ** 2)))
            # Gradients of MSE with respect to weight and bias
            self.weight -= lr * 2 * np.mean(error * x)
            self.bias -= lr * 2 * np.mean(error)
        return losses
