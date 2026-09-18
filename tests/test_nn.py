"""MLP forward/backward sanity checks (seeded, CPU-only, no torch needed)."""

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micrograd"))

from engine import Value
from nn import MLP, Layer, Neuron


class TestNeuron(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_forward_is_wx_plus_b(self):
        n = Neuron(3)
        for wi, xi in zip(n.w, [1.0, 1.0, 1.0]):
            wi.data = xi
        n.b.data = 0.0
        out = n([1.0, 1.0, 1.0])
        self.assertAlmostEqual(out.data, math.tanh(3.0), places=6)

    def test_parameters_count(self):
        n = Neuron(4)
        self.assertEqual(len(n.parameters()), 5)  # 4 weights + bias


class TestMLP(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_single_value_output_for_1_neuron_head(self):
        mlp = MLP(3, [4, 1])
        out = mlp([0.5, -0.5, 1.0])
        self.assertIsInstance(out, Value)

    def test_backward_fills_all_grads(self):
        mlp = MLP(2, [3, 1])
        out = mlp([0.5, -1.0])
        out.backward()
        grads = [p.grad for p in mlp.parameters()]
        self.assertEqual(len(grads), 3 * (2 + 1) + 1 * (3 + 1))  # hidden + output
        self.assertTrue(any(g != 0.0 for g in grads),
                         "a tanh MLP backward pass must reach some parameter")

    def test_zero_grad(self):
        mlp = MLP(2, [3, 1])
        mlp([0.5, -1.0]).backward()
        mlp.zero_grad()
        self.assertTrue(all(p.grad == 0.0 for p in mlp.parameters()))

    def test_gradient_step_reduces_loss(self):
        random.seed(1)
        mlp = MLP(2, [4, 1])
        data = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(8)]

        def sse():
            # target is 0.0 everywhere -> loss is sum of squared predictions
            return sum((p * p) for p in [mlp([x, y]) for (x, y) in data])

        before = sum((mlp([x, y]).data) ** 2 for (x, y) in data)
        for _ in range(10):
            out = sse()
            mlp.zero_grad()
            out.backward()
            for p in mlp.parameters():
                p.data += -0.05 * p.grad
        after = sum((mlp([x, y]).data) ** 2 for (x, y) in data)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
