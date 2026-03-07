# ML From Scratch

Building language models from the ground up.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/pip install numpy matplotlib jupyter
```

## Phase 1 — Foundations

Understanding backpropagation, embeddings, and attention from scratch.

| Script | What it teaches |
|--------|----------------|
| `micrograd/engine.py` | Scalar autograd engine (~100 lines) |
| `micrograd/nn.py` | Neuron / Layer / MLP built on engine |
| `micrograd/train.py` | Train on XOR, watch loss fall |
| `makemore/01_bigram.py` | Count-based vs neural LM, NLL loss |
| `makemore/02_mlp.py` | Embeddings, context window, train/val/test split |
| `makemore/03_transformer.py` | Full transformer: attention, residuals, positional embeddings |

### Run

```bash
.venv/Scripts/python micrograd/train.py
.venv/Scripts/python makemore/01_bigram.py
.venv/Scripts/python makemore/02_mlp.py
.venv/Scripts/python makemore/03_transformer.py
```

## Phase 2 — Real Text Generation (coming)

## Phase 3 — Daily Use Case (coming)
