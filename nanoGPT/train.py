"""
nanoGPT training loop — Phase 2
=================================
What this adds over Phase 1:

  1. Mixed precision (bfloat16) — halves VRAM, same quality
  2. Gradient accumulation — simulate large batches without OOM
  3. Gradient clipping — prevents exploding gradients in deep nets
  4. Cosine LR schedule — warmup then decay, better than fixed LR
  5. AdamW with weight decay — regularization via optimizer
  6. Checkpointing — save best model, resume training
  7. Throughput logging — tokens/sec tells you if GPU is being used well

Config for RTX 3050 4GB: ~10M params, trains in ~15 minutes.

Run: python train.py
"""

import os
import sys
import math
import time
import json
import numpy as np
import torch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from nanoGPT.model import GPT

# ------------------------------------------------------------------ #
#  Config                                                               #
# ------------------------------------------------------------------ #

@dataclass
class GPTConfig:
    # Model
    vocab_size:  int   = 65       # will be updated from vocab.json
    block_size:  int   = 256      # context length
    n_layer:     int   = 6
    n_head:      int   = 6
    n_embd:      int   = 384
    dropout:     float = 0.1
    bias:        bool  = False    # no bias → fewer params, usually fine

@dataclass
class TrainConfig:
    # Data
    data_dir:    str   = 'nanoGPT'

    # Training
    max_iters:   int   = 5000
    batch_size:  int   = 4       # micro batch (minimum for 4GB VRAM)
    grad_accum:  int   = 16      # effective batch = 4 * 16 = 64

    # Optimizer
    lr_max:      float = 3e-4
    lr_min:      float = 3e-5    # cosine decay floor
    warmup_iters:int   = 200
    weight_decay:float = 0.1
    betas:       tuple = (0.9, 0.95)
    grad_clip:   float = 1.0

    # Eval
    eval_interval: int = 500
    eval_iters:    int = 100

    # Logging
    log_interval:  int = 50

    # Checkpointing
    out_dir:     str   = 'nanoGPT/out'

# ------------------------------------------------------------------ #
#  Setup                                                                #
# ------------------------------------------------------------------ #

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.bfloat16   # RTX 3050 supports bf16
torch.manual_seed(42)

cfg   = GPTConfig()
tcfg  = TrainConfig()
os.makedirs(tcfg.out_dir, exist_ok=True)

# Load vocab
with open(os.path.join(tcfg.data_dir, 'vocab.json')) as f:
    vocab = json.load(f)
cfg.vocab_size = vocab['vocab_size']
itos = {int(k): v for k, v in vocab['itos'].items()}

# Load data
train_data = np.fromfile(os.path.join(tcfg.data_dir, 'train.bin'), dtype=np.uint16)
val_data   = np.fromfile(os.path.join(tcfg.data_dir, 'val.bin'),   dtype=np.uint16)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - cfg.block_size, (tcfg.batch_size,))
    x  = torch.stack([torch.from_numpy(data[i  :i+cfg.block_size  ].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1:i+cfg.block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ------------------------------------------------------------------ #
#  Model                                                                #
# ------------------------------------------------------------------ #

model = GPT(cfg).to(device)
model = model.to(dtype)
print(f"Model: {model.num_params()/1e6:.2f}M parameters")
print(f"Device: {device} | dtype: {dtype}")
print(f"Effective batch size: {tcfg.batch_size * tcfg.grad_accum}")
print(f"Config: {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}D block={cfg.block_size}")
print()

optimizer = model.configure_optimizer(tcfg.lr_max, tcfg.weight_decay, tcfg.betas)
scaler    = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))  # bf16 doesn't need scaling

# ------------------------------------------------------------------ #
#  LR Schedule: linear warmup → cosine decay                           #
# ------------------------------------------------------------------ #

def get_lr(it):
    # Warmup
    if it < tcfg.warmup_iters:
        return tcfg.lr_max * it / tcfg.warmup_iters
    # Cosine decay
    decay_ratio = (it - tcfg.warmup_iters) / (tcfg.max_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return tcfg.lr_min + coeff * (tcfg.lr_max - tcfg.lr_min)

# ------------------------------------------------------------------ #
#  Eval                                                                 #
# ------------------------------------------------------------------ #

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    torch.cuda.empty_cache()
    return out

# ------------------------------------------------------------------ #
#  Training loop                                                        #
# ------------------------------------------------------------------ #

best_val_loss = float('inf')
t0 = time.time()
X, Y = get_batch('train')

for it in range(tcfg.max_iters):
    # LR update
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Eval checkpoint
    if it % tcfg.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | lr: {lr:.2e}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model': model.state_dict(),
                'cfg': cfg,
                'val_loss': best_val_loss,
                'iter': it,
            }, os.path.join(tcfg.out_dir, 'best.pt'))

    # Gradient accumulation loop
    optimizer.zero_grad()
    for micro_step in range(tcfg.grad_accum):
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(X, Y)
            loss = loss / tcfg.grad_accum   # scale loss by accum steps
        scaler.scale(loss).backward()
        X, Y = get_batch('train')   # fetch next batch after backward frees activations

    # Gradient clipping — cap gradient norm to prevent explosions
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    # Logging
    if it % tcfg.log_interval == 0:
        dt = time.time() - t0
        tokens_per_sec = (tcfg.log_interval * tcfg.batch_size * tcfg.grad_accum * cfg.block_size) / dt
        print(f"  iter {it:5d} | loss: {loss.item()*tcfg.grad_accum:.4f} | "
              f"{tokens_per_sec/1e3:.1f}k tok/s | grad_norm: {grad_norm:.2f}")
        t0 = time.time()

print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {tcfg.out_dir}/best.pt")

# ------------------------------------------------------------------ #
#  Quick sample at the end                                              #
# ------------------------------------------------------------------ #

print("\n--- Sample (temperature=0.8, top_k=40) ---\n")
model.eval()
checkpoint = torch.load(os.path.join(tcfg.out_dir, 'best.pt'), weights_only=False)
model.load_state_dict(checkpoint['model'])

start = "\nFIRST CITIZEN:\n"
start_ids = [vocab['stoi'][c] for c in start]
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=dtype):
        y = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=40)

print(''.join(itos[i] for i in y[0].tolist()))
