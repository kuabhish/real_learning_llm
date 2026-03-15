import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, urllib.request, time

# ── DEVICE ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── CONFIG — tuned for ~30-60 min on Mac GPU ──────────────────────────────────
VOCAB_SIZE  = None   # set after building vocab
SEQ_LEN     = 128
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 4
D_FF        = 1024
BATCH_SIZE  = 64
LR          = 3e-4
STEPS       = 10000
EVAL_EVERY  = 500
EVAL_STEPS  = 50

# ── DATA — auto download TinyShakespeare (1MB, good variety) ──────────────────
# Want more data? swap the URL for:
# "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# or point DATA_PATH at any .txt file on your machine
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data.txt"

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Done.")

with open(DATA_PATH, 'r') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# ── TOKENIZER — character level ───────────────────────────────────────────────
chars      = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for c, i in stoi.items()}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda ids: ''.join(itos[i] for i in ids)

data  = torch.tensor(encode(text), dtype=torch.long)
n     = int(0.9 * len(data))
train = data[:n].to(device)
val   = data[n:].to(device)

print(f"Vocab size:       {VOCAB_SIZE}")
print(f"Training tokens:  {len(train):,}")
print(f"Validation tokens:{len(val):,}")

# ── BATCH SAMPLER ─────────────────────────────────────────────────────────────
def get_batch(split):
    d  = train if split == 'train' else val
    ix = torch.randint(len(d) - SEQ_LEN, (BATCH_SIZE,))
    x  = torch.stack([d[i:i+SEQ_LEN]     for i in ix])
    y  = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix])
    return x, y

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# Each class has a clear comment showing WHERE to plug in architecture changes
# ══════════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ── [HOOK A] ── want to change HOW attention scores are computed?
        # e.g. replace dot-product with something else, add relative position bias,
        # change the score function entirely — do it here
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)

        # ── [HOOK B] ── want to change HOW values are aggregated?
        # e.g. use a different mixing operation instead of weighted sum
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        # ── [HOOK C] ── want to change the "thinking space"?
        # e.g. add gating (like in LLaMA), use a different activation,
        # add an extra layer, make it sparse — do it here
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn  = SelfAttention(d_model, n_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # ── [HOOK D] ── want to change the BLOCK structure itself?
        # e.g. add a second attention layer, add cross-attention,
        # change the order of operations, add a new module entirely — here
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)

        # ── [HOOK E] ── want to change the STACK of blocks?
        # e.g. alternate between different block types, add shared layers,
        # use a mixture of experts — here
        self.blocks = nn.Sequential(*[
            TransformerBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)
        ])

        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids, targets=None):
        B, T   = token_ids.shape
        tok    = self.token_emb(token_ids)
        pos    = self.pos_emb(torch.arange(T, device=token_ids.device))
        x      = tok + pos

        x      = self.blocks(x)
        x      = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=300, temperature=0.8):
        # temperature < 1.0 = more focused, > 1.0 = more creative
        self.eval()
        ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            ids_crop      = ids[:, -SEQ_LEN:]
            logits, _     = self(ids_crop)
            logits        = logits[:, -1, :] / temperature
            probs         = F.softmax(logits, dim=-1)
            next_id       = torch.multinomial(probs, num_samples=1)
            ids           = torch.cat([ids, next_id], dim=1)
        self.train()
        return decode(ids[0].tolist())

# ── TRAINING ──────────────────────────────────────────────────────────────────
model     = TinyLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
n_params  = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")
print(f"Training for {STEPS} steps, batch size {BATCH_SIZE}, seq len {SEQ_LEN}\n")

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb  = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

start = time.time()
for step in range(STEPS):
    xb, yb    = get_batch('train')
    _, loss   = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping — stops exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        losses  = estimate_loss()
        elapsed = (time.time() - start) / 60
        print(f"step {step:5d}  |  train: {losses['train']:.4f}  val: {losses['val']:.4f}  |  {elapsed:.1f} min")
        # sneak peek at generation every 2000 steps
        if step > 0 and step % 2000 == 0:
            print("\n--- sample ---")
            print(model.generate("ROMEO:", max_new_tokens=150))
            print("--------------\n")

# ── FINAL GENERATION ──────────────────────────────────────────────────────────
print("\n=== Final generated text ===\n")
print(model.generate("ROMEO:", max_new_tokens=400))

# ── SAVE ──────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "tinyllm_v2.pt")
print("\nModel saved to tinyllm_v2.pt")
