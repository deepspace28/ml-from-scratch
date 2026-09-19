"""
A tiny neural network library built on top of engine.py.
Shows how neurons, layers, and MLPs are just compositions of Value ops.
"""

import random
from engine import Value


class Neuron:
    def __init__(self, n_inputs, activation='tanh'):
        # weights and bias are just Values — they'll accumulate gradients
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        # w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        return act  # linear

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize()}Neuron({len(self.w)})"


class Layer:
    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP:
    """Multi-layer perceptron: stack of layers.

    Hidden layers use tanh. The output layer defaults to tanh too, but
    pass output_activation='linear' for regression (unbounded targets)
    or 'sigmoid' for binary classification (logistic output).
    """
    def __init__(self, n_inputs, layer_sizes, output_activation='tanh'):
        sizes = [n_inputs] + layer_sizes
        self.layers = [
            Layer(sizes[i], sizes[i+1],
                  activation='tanh' if i < len(layer_sizes)-1 else output_activation)
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"MLP({', '.join(str(l) for l in self.layers)})"
