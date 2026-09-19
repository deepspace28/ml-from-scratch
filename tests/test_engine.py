"""Numerical checks for the scalar autograd engine.

Every derivative is verified against a central finite difference, and
backprop through a compound expression is verified against a hand-computed
result — so a regression in any single op fails loudly.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micrograd"))

from engine import Value


def _grad_of(f, x, eps=1e-5):
    """Central finite difference of scalar expression f (as a Value) at x."""
    return (f(Value(x + eps)).data - f(Value(x - eps)).data) / (2 * eps)


def _assert_backward_matches_finite_diff(case, expr, points):
    for x in points:
        out = expr(Value(x))
        out.backward()
        fd = _grad_of(expr, x)
        case.assertAlmostEqual(xv.grad, fd, places=4)
        xv.grad = 0.0


xv = Value(0.0)  # scratch variable reused by the finite-difference helper


class TestValueForward(unittest.TestCase):
    def test_add_mul_div_pow(self):
        a, b = Value(2.0), Value(3.0)
        self.assertEqual((a + b).data, 5.0)
        self.assertEqual((a * b).data, 6.0)
        self.assertEqual((a / b).data, 2.0 / 3.0)
        self.assertEqual((a**3).data, 8.0)
        self.assertEqual((b - a).data, 1.0)
        self.assertEqual((-a).data, -2.0)

    def test_scalar_coercion_and_r_ops(self):
        self.assertEqual((Value(1.0) + 2).data, 3.0)
        self.assertEqual((2 + Value(1.0)).data, 3.0)
        self.assertEqual((3 / Value(2.0)).data, 1.5)
        self.assertEqual((Value(3.0) - 1).data, 2.0)

    def test_unary_activations(self):
        x = Value(0.5)
        self.assertEqual(x.tanh().data, math.tanh(0.5))
        self.assertEqual(x.exp().data, math.exp(0.5))
        self.assertEqual(x.log().data, math.log(0.5))
        self.assertEqual(Value(-1.0).relu().data, 0.0)


class TestBackward(unittest.TestCase):
    def _fresh(self, x):
        return Value(x)

    def test_quadratic_matches_finite_diff(self):
        val = self._fresh(2.0)
        (val * val).backward()
        self.assertAlmostEqual(
            val.grad, _grad_of(lambda v: v * v, 2.0), places=4
        )

    def test_tanh_matches_finite_diff(self):
        val = self._fresh(0.7)
        out = val.tanh()
        out.backward()
        self.assertAlmostEqual(val.grad, _grad_of(lambda v: v.tanh(), 0.7), places=4)

    def test_exp_matches_finite_diff(self):
        val = self._fresh(1.2)
        val.exp().backward()
        self.assertAlmostEqual(val.grad, _grad_of(lambda v: v.exp(), 1.2), places=4)

    def test_log_matches_finite_diff(self):
        val = self._fresh(3.0)
        val.log().backward()
        self.assertAlmostEqual(val.grad, _grad_of(lambda v: v.log(), 3.0), places=4)

    def test_sigmoid_matches_finite_diff(self):
        val = self._fresh(0.8)
        val.sigmoid().backward()
        self.assertAlmostEqual(
            val.grad, _grad_of(lambda v: v.sigmoid(), 0.8), places=4
        )

    def test_sigmoid_forward_extremes(self):
        self.assertAlmostEqual(Value(0.0).sigmoid().data, 0.5, places=6)
        self.assertGreater(Value(50.0).sigmoid().data, 0.9999)
        self.assertLess(Value(-50.0).sigmoid().data, 0.0001)

    def test_neg_pow_matches_finite_diff(self):
        val = self._fresh(4.0)
        (val**-1).backward()
        self.assertAlmostEqual(
            val.grad, _grad_of(lambda v: v**-1, 4.0), places=4
        )

    def test_relu_grad_is_zero_below_threshold(self):
        val = self._fresh(-0.5)
        val.relu().backward()
        self.assertEqual(val.grad, 0.0)

        val = self._fresh(0.5)
        val.relu().backward()
        self.assertEqual(val.grad, 1.0)

    def test_compound_expression_exact(self):
        # f(x) = tanh(x * 3 + 1); d/dx = 3 * (1 - tanh^2(x*3+1))
        val = self._fresh(0.4)
        out = (val * 3 + 1).tanh()
        out.backward()
        expected = 3 * (1 - math.tanh(3 * 0.4 + 1) ** 2)
        self.assertAlmostEqual(val.grad, expected, places=6)

    def test_gradients_accumulate_across_scalars(self):
        # y = x + x  ->  dy/dx = 2
        val = self._fresh(1.0)
        (val + val).backward()
        self.assertEqual(val.grad, 2.0)

    def test_repeated_use_via_product(self):
        # y = x * x * x -> dy/dx = 3x^2
        val = self._fresh(2.0)
        y = val * val * val
        y.backward()
        self.assertEqual(y.data, 8.0)
        self.assertAlmostEqual(val.grad, 3 * 4.0, places=6)

    def test_division_backward(self):
        # y = a / b -> d/da = 1/b, d/db = -a/b^2
        a, b = self._fresh(6.0), self._fresh(3.0)
        (a / b).backward()
        self.assertAlmostEqual(a.grad, 1 / 3.0, places=6)
        self.assertAlmostEqual(b.grad, -6.0 / 9.0, places=6)


class TestEnginePlumbing(unittest.TestCase):
    def test_grad_starts_at_zero(self):
        self.assertEqual(Value(1.0).grad, 0.0)

    def test_labels_and_ops_recorded(self):
        a, b = Value(1.0, label="a"), Value(2.0, label="b")
        c = a * b
        self.assertEqual(c._op, "*")
        self.assertEqual(a.label, "a")
        self.assertIn(a, c._prev)
        self.assertIn(b, c._prev)

    def test_repr(self):
        self.assertIn("data=1.0000", repr(Value(1.0)))


if __name__ == "__main__":
    unittest.main()
