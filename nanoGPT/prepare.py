"""
Tokenize Shakespeare and save as binary files for fast training.
Character-level tokenizer — same as Phase 1, but saved to disk
so the training loop doesn't re-read the text file every run.

Run once: python prepare.py
"""

import numpy as np
import os

data = open('nanoGPT/shakespeare.txt', 'r').read()
print(f"Dataset: {len(data):,} characters")

chars = sorted(set(data))
vocab_size = len(chars)
print(f"Vocabulary: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Save vocab for use in training + generation
import json
with open('nanoGPT/vocab.json', 'w') as f:
    json.dump({'stoi': stoi, 'itos': {str(k): v for k, v in itos.items()}, 'vocab_size': vocab_size}, f)

# Encode full dataset
data_ids = np.array([stoi[c] for c in data], dtype=np.uint16)

# 90/10 train/val split
n = int(0.9 * len(data_ids))
train_ids = data_ids[:n]
val_ids   = data_ids[n:]

train_ids.tofile('nanoGPT/train.bin')
val_ids.tofile('nanoGPT/val.bin')

print(f"Train tokens: {len(train_ids):,}  -> nanoGPT/train.bin")
print(f"Val tokens:   {len(val_ids):,}  -> nanoGPT/val.bin")
print(f"Vocab size:   {vocab_size}  -> nanoGPT/vocab.json")
