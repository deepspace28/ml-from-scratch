"""
makemore Step 1: Bigram Language Model
=======================================
Teaches:
- What a language model actually is (predict next token given context)
- Counting statistics vs learned probabilities
- Negative log likelihood as loss
- Why smoothing matters

A bigram model only looks at the PREVIOUS character to predict the NEXT one.
It's the simplest possible language model.

Run: python 01_bigram.py
"""

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ #
#  Load data                                                            #
# ------------------------------------------------------------------ #
words = open('makemore/names.txt').read().splitlines()
print(f"Dataset: {len(words)} names, e.g. {words[:5]}")

# Build character vocabulary
chars = sorted(set(''.join(words)))
# '.' is our special start/end token (index 0)
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)
print(f"Vocabulary: {vocab_size} characters")

# ------------------------------------------------------------------ #
#  Part A: Count-based bigram model                                    #
# ------------------------------------------------------------------ #
# N[i,j] = how many times character j follows character i
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# Convert counts to probabilities (add 1 for smoothing — avoids log(0))
P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)  # normalize each row

# Sample from the model
print("\n--- Count-based bigram samples ---")
g = torch.Generator().manual_seed(42)
for _ in range(10):
    out = []
    ix = 0   # start token
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))

# Evaluate: negative log likelihood
log_likelihood = 0.0
n = 0
for w in words[:5]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob).item()
        n += 1
nll = -log_likelihood / n
print(f"\nNegative log likelihood (lower = better): {nll:.4f}")

# ------------------------------------------------------------------ #
#  Part B: Neural bigram model (same thing, but learned via gradient)  #
# ------------------------------------------------------------------ #
print("\n--- Neural bigram (same model, learned via gradient descent) ---")

# Build dataset: (input_char, target_char) pairs
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"Training pairs: {xs.shape[0]}")

# W is the weight matrix — equivalent to the count table, but learned
g = torch.Generator().manual_seed(42)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

# Training
for step in range(200):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=vocab_size).float()  # one-hot encode input
    logits = xenc @ W                                      # (N, 27) @ (27, 27) → (N, 27)
    probs = F.softmax(logits, dim=1)                       # convert to probabilities
    loss = F.nll_loss(torch.log(probs), ys)                # negative log likelihood

    # Backward
    W.grad = None
    loss.backward()

    # Update
    W.data -= 50 * W.grad

    if step % 40 == 0 or step == 199:
        print(f"step {step:3d} | loss: {loss.item():.4f}")

# Sample from neural model
print("\n--- Neural bigram samples ---")
g = torch.Generator().manual_seed(42)
for _ in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))

print("""
KEY INSIGHT:
  The neural model learns the same distribution as counting, but via gradient descent.
  This is important because it GENERALIZES: next we'll give the model MORE context
  (not just 1 character) and the gradient approach scales — counting doesn't.
""")
