"""
makemore Step 3: Transformer Language Model
============================================
Teaches:
- Self-attention: how tokens look at each other
- Multi-head attention: multiple "views" in parallel
- Causal masking: prevent looking at future tokens
- Residual connections + LayerNorm: stable deep training
- Positional embeddings: inject order information
- How this directly scales to GPT

This is the architecture. After this, nanoGPT is just a larger version.

Run: python 03_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

# ------------------------------------------------------------------ #
#  Data                                                                 #
# ------------------------------------------------------------------ #
words = open('makemore/names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

BLOCK_SIZE = 8   # context length

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

import random
random.shuffle(words)
n = int(0.9 * len(words))
Xtr, Ytr   = build_dataset(words[:n])
Xval, Yval = build_dataset(words[n:])

# ------------------------------------------------------------------ #
#  Architecture                                                         #
# ------------------------------------------------------------------ #

class Head(nn.Module):
    """Single self-attention head."""
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # causal mask: lower triangular — can't look at future
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Attention scores: how much does each position attend to each other
        # Scale by sqrt(head_size) to prevent softmax from saturating
        wei = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)               # normalize
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v     # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Multiple heads in parallel, outputs concatenated."""
    def __init__(self, n_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)   # projection back to residual stream

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Per-position MLP — 4x expansion then back down."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention + feedforward with residual connections."""
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa  = MultiHeadAttention(n_heads, head_size, n_embd, block_size)
        self.ff  = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-norm: normalize BEFORE attention (modern convention)
        x = x + self.sa(self.ln1(x))   # residual: x + attention(x)
        x = x + self.ff(self.ln2(x))   # residual: x + ffn(x)
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_heads=4, n_layers=4, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.head   = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                          # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # targets shape (B,) → predict from last position only
            loss = F.cross_entropy(logits[:, -1, :], targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# ------------------------------------------------------------------ #
#  Train                                                                #
# ------------------------------------------------------------------ #
model = CharTransformer(vocab_size).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
BATCH_SIZE = 64
STEPS = 5000

for step in range(STEPS):
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
    x, y = Xtr[ix].to(device), Ytr[ix].to(device)
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            ix = torch.randint(0, Xval.shape[0], (512,))
            _, val_loss = model(Xval[ix].to(device), Yval[ix].to(device))
        print(f"step {step:5d} | train: {loss.item():.4f} | val: {val_loss.item():.4f}")
        model.train()

# ------------------------------------------------------------------ #
#  Sample                                                               #
# ------------------------------------------------------------------ #
print("\n--- Transformer samples ---")
model.eval()
g = torch.Generator().manual_seed(42)
for _ in range(15):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=20)[0].tolist()
    name = ''.join(itos.get(i, '') for i in generated if i != 0)
    print(name)

print("""
YOU JUST BUILT A TRANSFORMER from scratch.
Every component in GPT-2/3/4 is the same — just bigger:
  - More layers (n_layers: 4 to 96)
  - Bigger embeddings (n_embd: 64 to 12288)
  - More heads (n_heads: 4 to 96)
  - More data (32K names to 100B tokens)
Next: nanoGPT — same architecture, real dataset, real output.
""")
