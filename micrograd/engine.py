"""
micrograd: scalar-valued autograd engine
Teaches: how backpropagation actually works, one operation at a time.

Every Value wraps a single float. Operations (+, *, tanh, etc.) record
how to compute the gradient of the output w.r.t. the inputs (_backward).
.backward() walks the computation graph in reverse (chain rule).
"""

import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0          # starts at zero — no gradient yet
        self._backward = lambda: None   # filled in by each op
        self._prev = set(_children)
        self._op = _op           # what created this node (for viz)
        self.label = label

    # ------------------------------------------------------------------ #
    #  Forward operations                                                   #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule: d(a*b)/da = b
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh)/dx = 1 - tanh^2(x)
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Backprop                                                             #
    # ------------------------------------------------------------------ #

    def backward(self):
        """Topological sort then call _backward in reverse order."""
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0   # d(output)/d(output) = 1
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ #
    #  Convenience                                                          #
    # ------------------------------------------------------------------ #

    def __neg__(self):         return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
