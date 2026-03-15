import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match your training script exactly
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = "mindpalace_v1.pt"
DATA_FILE  = "data.txt"
SEQ_LEN    = 128
D_MODEL    = 256
N_HEADS    = 8
D_FF       = 1024

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    text = f.read()
chars      = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for c, i in stoi.items()}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda ids: ''.join(itos[i] for i in ids)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION (identical to training)
# ─────────────────────────────────────────────────────────────────────────────
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
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        out    = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.net(x)

class Room(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, room_id):
        super().__init__()
        self.room_id = room_id
        self.attn    = SelfAttention(d_model, n_heads)
        self.ff      = FeedForward(d_model, d_ff)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.summary = nn.Parameter(torch.randn(d_model) * 0.01)
        self.register_buffer('heat', torch.tensor(1.0))
    def forward(self, x, gate_score):
        residual = x
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return residual + gate_score.unsqueeze(-1).unsqueeze(-1) * (x - residual)

class MindPalaceRouter(nn.Module):
    def __init__(self, d_model, n_rooms):
        super().__init__()
        self.context_proj = nn.Linear(d_model, d_model)
        self.adjacency    = nn.Parameter(torch.eye(n_rooms) + torch.randn(n_rooms, n_rooms) * 0.1)
        self.warp_proj    = nn.Linear(d_model, n_rooms * n_rooms)
    def forward(self, x, summaries, hard=False):
        B, n  = x.shape[0], summaries.shape[0]
        ctx   = self.context_proj(x.mean(dim=1))                          # (B, D)
        warp  = self.warp_proj(ctx).view(B, n, n) * 0.1
        adj   = torch.softmax(self.adjacency.unsqueeze(0) + warp, dim=-1) # (B, n, n)
        raw   = (ctx.unsqueeze(1) * summaries.unsqueeze(0).expand(B,-1,-1)).sum(-1) # (B, n)
        gates = torch.sigmoid((adj * raw.unsqueeze(1)).sum(-1))            # (B, n)
        return (gates > 0.5).float() if hard else gates

class MindPalace(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_rooms):
        super().__init__()
        self.rooms  = nn.ModuleList([Room(d_model, n_heads, d_ff, i) for i in range(n_rooms)])
        self.router = MindPalaceRouter(d_model, n_rooms)
    def forward(self, x, hard=False):
        summaries = torch.stack([r.summary for r in self.rooms])
        gates     = self.router(x, summaries, hard=hard)   # (B, n_rooms)
        for i, room in enumerate(self.rooms):
            x = room(x, gates[:, i])
        return x, gates

class MindPalaceLLM(nn.Module):
    def __init__(self, n_rooms):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)
        self.palace    = MindPalace(D_MODEL, N_HEADS, D_FF, n_rooms)
        self.norm      = nn.LayerNorm(D_MODEL)
        self.head      = nn.Linear(D_MODEL, VOCAB_SIZE)
    def forward(self, ids, hard=False):
        B, T  = ids.shape
        x     = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        x, g  = self.palace(x, hard=hard)
        return self.head(self.norm(x)), g

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
ckpt    = torch.load(CHECKPOINT, map_location=device)
n_rooms = len({k.split('.')[2] for k in ckpt if k.startswith('palace.rooms.')})
model   = MindPalaceLLM(n_rooms).to(device)
model.load_state_dict(ckpt)
model.eval()
print(f"loaded  |  rooms: {n_rooms}  |  params: {sum(p.numel() for p in model.parameters()):,}\n")

# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt, max_tokens=300, temp=0.8, hard=False):
    ids      = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    all_gates = []
    t0        = time.time()
    for _ in range(max_tokens):
        logits, gates = model(ids[:, -SEQ_LEN:], hard=hard)
        all_gates.append(gates.squeeze(0).cpu())
        probs   = F.softmax(logits[:, -1, :] / temp, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids     = torch.cat([ids, next_id], dim=1)
    elapsed = time.time() - t0
    text    = decode(ids[0].tolist())
    gates   = torch.stack(all_gates)   # (max_tokens, n_rooms)
    return text, gates, elapsed

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def show_gates(gates, label=""):
    avg = gates.mean(dim=0)   # (n_rooms,)
    print(f"  {DIM}room activations {label}{RESET}")
    for i, g in enumerate(avg):
        filled = int(g.item() * 24)
        bar    = '█' * filled + '░' * (24 - filled)
        heat   = model.palace.rooms[i].heat.item()
        print(f"  room {i}  {bar}  avg={g.item():.2f}  heat={heat:.2f}")
    active = (avg > 0.5).sum().item()
    print(f"  {DIM}avg rooms active: {active}/{n_rooms}{RESET}\n")

def run(prompt, label, max_tokens=300, temp=0.8, hard=False):
    mode = "hard gates" if hard else "soft gates"
    print(f"{BOLD}{'─'*62}{RESET}")
    print(f"{BOLD}  {label}{RESET}  {DIM}[{mode}  temp={temp}]{RESET}")
    print(f"  prompt: {CYAN}{prompt}{RESET}\n")
    out, gates, elapsed = generate(prompt, max_tokens=max_tokens, temp=temp, hard=hard)
    generated = out[len(prompt):]
    # word-wrap at 60 chars for readability
    words, line = generated.split(' '), ''
    wrapped = []
    for w in words:
        if len(line) + len(w) + 1 > 60:
            wrapped.append(line)
            line = w
        else:
            line = (line + ' ' + w).strip()
    if line: wrapped.append(line)
    for ln in wrapped:
        print(f"  {GREEN}{ln}{RESET}")
    print(f"\n  {DIM}{max_tokens} tokens  {elapsed:.1f}s  ({max_tokens/elapsed:.0f} tok/s){RESET}")
    show_gates(gates)

def compare(prompt, max_tokens=200):
    print(f"{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  SOFT vs HARD  —  \"{prompt}\"{RESET}\n")
    out_s, g_s, t_s = generate(prompt, max_tokens=max_tokens, hard=False)
    out_h, g_h, t_h = generate(prompt, max_tokens=max_tokens, hard=True)
    print(f"  {CYAN}SOFT  ({t_s:.1f}s){RESET}")
    print(f"  {GREEN}{out_s[len(prompt):][:200]}{RESET}\n")
    print(f"  {YELLOW}HARD  ({t_h:.1f}s)  speedup: {t_s/t_h:.2f}x{RESET}")
    print(f"  {GREEN}{out_h[len(prompt):][:200]}{RESET}\n")
    show_gates(g_s, label="soft")
    show_gates(g_h, label="hard")

# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'█'*62}{RESET}")
print(f"{BOLD}  MIND PALACE — TEST SUITE{RESET}")
print(f"{BOLD}{'█'*62}{RESET}\n")

# room health
print(f"{BOLD}  room health:{RESET}")
for r in model.palace.rooms:
    b = '█' * int(r.heat.item() * 20) + '░' * (20 - int(r.heat.item() * 20))
    print(f"  room {r.room_id}  [{b}]  {r.heat.item():.3f}")
print()

# ── 1. characters ─────────────────────────────────────────────────────────────
run("ROMEO:",         "Romeo",         max_tokens=300)
run("JULIET:",        "Juliet",        max_tokens=300)
run("HAMLET:",        "Hamlet",        max_tokens=300)
run("KING:",          "King",          max_tokens=300)
run("QUEEN:",         "Queen",         max_tokens=300)

# ── 2. famous lines ───────────────────────────────────────────────────────────
run("To be, or not to be,",   "Hamlet soliloquy",       max_tokens=250)
run("All the world's",        "As You Like It",          max_tokens=250)
run("Friends, Romans,",       "Julius Caesar",           max_tokens=250)
run("The quality of mercy",   "Merchant of Venice",      max_tokens=250)
run("What a piece of work",   "Hamlet on man",           max_tokens=250)

# ── 3. dialogue ───────────────────────────────────────────────────────────────
run("ROMEO:\nBut soft, what light",    "Romeo balcony",   max_tokens=300)
run("FIRST CITIZEN:\nBefore we",       "Citizen scene",   max_tokens=300)

# ── 4. temperature sweep ──────────────────────────────────────────────────────
run("ROMEO:", "temp 0.4 — focused",  max_tokens=200, temp=0.4)
run("ROMEO:", "temp 0.8 — balanced", max_tokens=200, temp=0.8)
run("ROMEO:", "temp 1.4 — creative", max_tokens=200, temp=1.4)

# ── 5. long generations ───────────────────────────────────────────────────────
run("To be, or not to be,", "long — 500 tokens", max_tokens=500)
run("ROMEO:",               "long — 500 tokens", max_tokens=500)

# ── 6. soft vs hard gate comparison ──────────────────────────────────────────
compare("ROMEO:",               max_tokens=200)
compare("To be, or not to be,", max_tokens=200)
compare("KING:",                max_tokens=200)

print(f"\n{BOLD}{'█'*62}{RESET}")
print(f"{BOLD}  DONE{RESET}")
print(f"{BOLD}{'█'*62}{RESET}\n")
