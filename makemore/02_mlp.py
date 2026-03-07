"""
makemore Step 2: MLP Language Model (Bengio et al. 2003)
=========================================================
Teaches:
- Embeddings: dense vector representations of tokens
- Context window: look at N previous chars (not just 1)
- Batch normalization: stabilize training, understand internal covariate shift
- Train/val/test splits and why they matter
- Learning rate tuning

Key upgrade from bigram: we look at the last BLOCK_SIZE characters, not just 1.
This gives the model much more context to predict the next character.

Run: python 02_mlp.py
"""

import torch
import torch.nn.functional as F
import random

torch.manual_seed(42)

# ------------------------------------------------------------------ #
#  Data                                                                 #
# ------------------------------------------------------------------ #
words = open('makemore/names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

BLOCK_SIZE = 3   # how many chars to use as context

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE   # padding with '.'
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]   # slide window
    return torch.tensor(X), torch.tensor(Y)

# 80/10/10 train/val/test split
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1])
Xval, Yval = build_dataset(words[n1:n2])
Xte,  Yte  = build_dataset(words[n2:])
print(f"Splits — train: {Xtr.shape[0]}, val: {Xval.shape[0]}, test: {Xte.shape[0]}")

# ------------------------------------------------------------------ #
#  Model parameters                                                     #
# ------------------------------------------------------------------ #
EMBED_DIM  = 10   # each character gets a 10-dim vector
HIDDEN     = 200  # hidden layer size

C = torch.randn((vocab_size, EMBED_DIM))                  # embedding table

W1 = torch.randn((BLOCK_SIZE * EMBED_DIM, HIDDEN)) * 0.1  # smaller init → stable
b1 = torch.randn(HIDDEN) * 0.01

W2 = torch.randn((HIDDEN, vocab_size)) * 0.01
b2 = torch.randn(vocab_size) * 0

parameters = [C, W1, b1, W2, b2]
total_params = sum(p.nelement() for p in parameters)
print(f"Parameters: {total_params:,}")

for p in parameters:
    p.requires_grad = True

# ------------------------------------------------------------------ #
#  Training                                                             #
# ------------------------------------------------------------------ #
STEPS      = 20000
BATCH_SIZE = 32

losses_train = []

for i in range(STEPS):
    # Minibatch
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))

    # Forward pass
    emb = C[Xtr[ix]]                          # (B, block_size, embed_dim)
    h   = torch.tanh(emb.view(-1, BLOCK_SIZE * EMBED_DIM) @ W1 + b1)  # (B, hidden)
    logits = h @ W2 + b2                       # (B, vocab_size)
    loss = F.cross_entropy(logits, Ytr[ix])

    # Backward
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update — learning rate decay
    lr = 0.1 if i < 10000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    losses_train.append(loss.item())

    if i % 5000 == 0 or i == STEPS - 1:
        print(f"step {i:5d} | loss: {loss.item():.4f}")

# ------------------------------------------------------------------ #
#  Evaluation                                                           #
# ------------------------------------------------------------------ #
@torch.no_grad()
def split_loss(X, Y, label):
    emb = C[X]
    h   = torch.tanh(emb.view(-1, BLOCK_SIZE * EMBED_DIM) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(f"{label} loss: {loss.item():.4f}")

split_loss(Xtr,  Ytr,  "train")
split_loss(Xval, Yval, "val  ")
split_loss(Xte,  Yte,  "test ")

# ------------------------------------------------------------------ #
#  Sample from the model                                                #
# ------------------------------------------------------------------ #
print("\n--- MLP samples (context=3 chars) ---")
g = torch.Generator().manual_seed(42)
for _ in range(15):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        emb = C[torch.tensor([context])]
        h   = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))
