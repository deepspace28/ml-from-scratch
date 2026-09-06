"""
Interactive generation from a trained checkpoint.

Usage:
  python generate.py                        # prompts you for input
  python generate.py --prompt "HAMLET:"     # direct prompt
  python generate.py --temp 0.6 --top_k 50 # control randomness
"""

import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from nanoGPT.model import GPT, GPTConfig

# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='nanoGPT/out/best.pt')
parser.add_argument('--prompt',     default=None)
parser.add_argument('--max_tokens', type=int,   default=500)
parser.add_argument('--temp',       type=float, default=0.8)
parser.add_argument('--top_k',      type=int,   default=40)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.bfloat16

# Load checkpoint
ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg   = ckpt['cfg']
model = GPT(cfg).to(device).to(dtype)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded checkpoint (val_loss={ckpt['val_loss']:.4f}, iter={ckpt['iter']})")

# Load vocab
with open('nanoGPT/vocab.json') as f:
    vocab = json.load(f)
stoi = vocab['stoi']
itos = {int(k): v for k, v in vocab['itos'].items()}

def generate(prompt, max_tokens, temp, top_k):
    ids = [stoi[c] for c in prompt if c in stoi]
    x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            y = model.generate(x, max_tokens, temperature=temp, top_k=top_k)
    return ''.join(itos[i] for i in y[0].tolist())

# ------------------------------------------------------------------ #
if args.prompt:
    print(generate(args.prompt, args.max_tokens, args.temp, args.top_k))
else:
    print(f"\nInteractive mode — type a prompt, Enter to generate, Ctrl+C to quit")
    print(f"(temperature={args.temp}, top_k={args.top_k})\n")
    while True:
        try:
            prompt = input("Prompt> ")
            if prompt:
                print(generate(prompt, args.max_tokens, args.temp, args.top_k))
                print()
        except KeyboardInterrupt:
            break
