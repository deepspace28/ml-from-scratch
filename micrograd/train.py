"""
Train a tiny MLP on XOR — the simplest non-linearly-separable problem.

Run this and watch loss fall. Every line of the training loop corresponds
directly to the theory: forward pass → loss → backward → gradient descent.

Usage: python train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from engine import Value
from nn import MLP

# ------------------------------------------------------------------ #
#  Dataset: XOR                                                         #
#  A single layer can't solve this — proves why depth matters.         #
# ------------------------------------------------------------------ #
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [0.0, 1.0, 1.0, 0.0]   # XOR outputs

# ------------------------------------------------------------------ #
#  Model: 2 inputs → 4 hidden (tanh) → 1 output (tanh)                #
# ------------------------------------------------------------------ #
model = MLP(2, [4, 1])
print(f"Model: {model}")
print(f"Parameters: {len(model.parameters())}")
print()

# ------------------------------------------------------------------ #
#  Training loop                                                        #
# ------------------------------------------------------------------ #
learning_rate = 0.1

for step in range(200):

    # --- Forward pass ---
    ypred = [model(x) for x in xs]

    # --- Loss: mean squared error ---
    loss = sum((yout - Value(ygt))**2 for ygt, yout in zip(ys, ypred))
    loss = loss * Value(1/len(ys))

    # --- Backward pass ---
    model.zero_grad()          # reset gradients from last step
    loss.backward()            # compute all gradients via chain rule

    # --- Gradient descent step ---
    for p in model.parameters():
        p.data -= learning_rate * p.grad   # move against gradient

    if step % 20 == 0 or step == 199:
        preds = [round(yp.data, 3) for yp in ypred]
        print(f"step {step:3d} | loss: {loss.data:.6f} | preds: {preds}")

print()
print("Target:", ys)
print("Final preds:", [round(model(x).data, 3) for x in xs])
print()
print("WHAT YOU JUST SAW:")
print("  - Forward pass: values flow forward through Value ops")
print("  - loss.backward(): chain rule walks graph in reverse, fills .grad")
print("  - Gradient descent: p.data -= lr * p.grad nudges weights toward lower loss")
