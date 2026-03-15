import torch, torch.nn as nn, torch.nn.functional as F, math

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("data.txt", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi  = {c: i for i, c in enumerate(chars)}
itos  = {i: c for c, i in stoi.items()}
enc   = lambda s: [stoi[c] for c in s if c in stoi]
dec   = lambda ids: ''.join(itos[i] for i in ids)

# ── model — names must exactly match the checkpoint ───────────────────────────
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
        Q = self.W_q(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        K = self.W_k(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        V = self.W_v(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        s = Q @ K.transpose(-2,-1) / math.sqrt(self.d_head)
        s = s.masked_fill(torch.triu(torch.ones(T,T,device=x.device),1).bool(), float('-inf'))
        return self.W_o((F.softmax(s,-1) @ V).transpose(1,2).contiguous().view(B,T,C))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model))
    def forward(self, x): return self.net(x)

class Room(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, room_id):
        super().__init__()
        self.room_id = room_id
        self.attn    = SelfAttention(d_model, n_heads)
        self.ff      = FeedForward(d_model, d_ff)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.summary         = nn.Parameter(torch.randn(d_model) * 0.01)
        self.confidence_head = nn.Linear(d_model, 1)
        self.register_buffer('heat',       torch.tensor(1.0))
        self.register_buffer('hot_streak', torch.tensor(0))
        self.register_buffer('age',        torch.tensor(0))
    def forward(self, x, gate_score):
        residual = x
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        out = residual + gate_score.unsqueeze(-1).unsqueeze(-1) * (x - residual)
        conf = torch.sigmoid(self.confidence_head(out.mean(dim=1)))
        return out, conf

class MindPalaceRouter(nn.Module):
    def __init__(self, d_model, n_rooms):
        super().__init__()
        self.context_proj = nn.Linear(d_model, d_model)
        self.gate_proj    = nn.Linear(d_model, n_rooms)
        self.adjacency    = nn.Parameter(torch.eye(n_rooms) + torch.randn(n_rooms,n_rooms)*0.1)
        self.warp_proj    = nn.Linear(d_model, n_rooms * n_rooms)
    def forward(self, x, summaries, hard=False):
        B, n = x.shape[0], summaries.shape[0]
        ctx  = self.context_proj(x.mean(1))
        adj  = torch.softmax(self.adjacency.unsqueeze(0) + self.warp_proj(ctx).view(B,n,n)*0.1, -1)
        raw  = (ctx.unsqueeze(1) * summaries.unsqueeze(0).expand(B,-1,-1)).sum(-1)
        g    = torch.sigmoid((self.gate_proj(ctx) + (adj * raw.unsqueeze(1)).sum(-1)) / 2.0)
        return (g > 0.5).float() if hard else g

class MindPalace(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_rooms):
        super().__init__()
        self.rooms  = nn.ModuleList([Room(d_model,n_heads,d_ff,i) for i in range(n_rooms)])
        self.router = MindPalaceRouter(d_model, n_rooms)
    def forward(self, x, hard=False):
        sums  = torch.stack([r.summary for r in self.rooms])
        gates = self.router(x, sums, hard)
        for i, room in enumerate(self.rooms):
            x, _ = room(x, gates[:,i])   # unpack (out, confidence)
        return x, gates

class MindPalaceLLM(nn.Module):
    def __init__(self, n_rooms):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB, 256)
        self.pos_emb   = nn.Embedding(128, 256)
        self.palace    = MindPalace(256, 8, 1024, n_rooms)
        self.norm      = nn.LayerNorm(256)
        self.head      = nn.Linear(256, VOCAB)
    def forward(self, ids, hard=False):
        B, T = ids.shape
        x    = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        x, g = self.palace(x, hard)
        return self.head(self.norm(x)), g, None   # 3rd value = n_visited (None at inference)

# ── load ──────────────────────────────────────────────────────────────────────
ckpt   = torch.load("mindpalace_v1.pt", map_location=device, weights_only=True)
n      = len({k.split('.')[2] for k in ckpt if k.startswith('palace.rooms.')})
model  = MindPalaceLLM(n).to(device)
model.load_state_dict(ckpt)
model.eval()
print(f"ready — {n} rooms\n")

# ── generate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def chat(prompt, tokens=400, temp=0.8):
    ids = torch.tensor(enc(prompt), dtype=torch.long).unsqueeze(0).to(device)
    print("\033[92m", end="", flush=True)
    for _ in range(tokens):
        logits, _, _ = model(ids[:, -128:])
        probs     = F.softmax(logits[:, -1, :] / temp, dim=-1)
        nxt       = torch.multinomial(probs, 1)
        ids       = torch.cat([ids, nxt], dim=1)
        print(dec([nxt.item()]), end="", flush=True)
    print("\033[0m\n")

# ── loop ──────────────────────────────────────────────────────────────────────
print("type anything. quit to exit. :temp 0.8  :tokens 400\n")
temp, tokens = 0.8, 400
while True:
    try:
        line = input("you › ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:                   continue
    if line == "quit":             break
    if line.startswith(":temp"):   temp   = float(line.split()[1]); print(f"temp → {temp}");     continue
    if line.startswith(":tokens"): tokens = int(line.split()[1]);   print(f"tokens → {tokens}"); continue
    print("model › ", end="", flush=True)
    chat(line, tokens=tokens, temp=temp)
    