# ML From Scratch

Building language models from the ground up.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/pip install numpy matplotlib jupyter
```

## Phase 1 â€” Foundations

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

## Phase 2 â€” Real Text Generation

Character-level GPT (nanoGPT recipe) trained from scratch on the complete works of Shakespeare.

| Script | What it does |
|--------|--------------|
| `nanoGPT/prepare.py` | Tokenize `shakespeare.txt` into `train.bin` / `val.bin` |
| `nanoGPT/model.py` | Decoder-only transformer: CausalSelfAttention, Block, GPT |
| `nanoGPT/train.py` | Training loop with AdamW, LR warmup + cosine decay, gradient clipping |
| `nanoGPT/generate.py` | Sample from the trained checkpoint (`--prompt`, `--temperature`, `--top-k`) |

### Run

```bash
.venv/Scripts/python nanoGPT/prepare.py
.venv/Scripts/python nanoGPT/train.py     # CUDA if available, CPU otherwise
.venv/Scripts/python nanoGPT/generate.py --prompt "HAMLET:"
```

## Phase 3 - Tokenizers

Real tokenizers, like GPT-2's, without the regex pre-split rule.

| Script | What it teaches |
|--------|----------------|
| `tokenizers/bpe.py` | Byte-pair encoding: train merges, encode/decode, compression ratio |

### Run

```bash
.venv/Scripts/python tokenizers/bpe.py --corpus makemore/names.txt --vocab-size 280
```

## Tests

Fast, torch-free checks for the pieces that don't need a GPU:

```bash
py -3.11 -m pytest tests/ -q
```

Covers the scalar autograd engine (every derivative verified against finite
differences), the MLP (backward reachability + a gradient step reducing loss),
and the BPE tokenizer (train/encode/decode round-trips incl. multi-byte UTF-8).
Runs on CI via .github/workflows/tests.yml.
