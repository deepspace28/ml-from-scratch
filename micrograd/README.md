# micrograd — Phase 1, Step 1

## What you're learning
- How backpropagation actually works
- Every operation records its own gradient rule (_backward)
- .backward() does topological sort → chain rule in reverse
- A neural net is just Value arithmetic

## Files
- `engine.py` — the autograd engine (~100 lines, read every line)
- `nn.py` — Neuron / Layer / MLP built on engine.py
- `train.py` — train on XOR, watch loss fall

## Run
```bash
cd /d/ml-from-scratch
.venv/Scripts/python micrograd/train.py
```

## Key questions to understand before moving to makemore
1. Why does zero_grad() matter? What breaks without it?
2. What is the topological sort doing in backward()?
3. Why can't a single linear layer solve XOR?
4. What does the chain rule look like for: z = (a + b) * c?

## Answers (check your understanding)
1. Grads accumulate — without zeroing, step 2 adds to step 1's gradients → wrong update
2. Ensures parent nodes compute grad before children (children need parent's .grad to be set)
3. XOR is not linearly separable — you need a non-linear hidden layer
4. dz/da = c, dz/db = c, dz/dc = (a+b) — each _backward implements exactly this
