import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, urllib.request, time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEQ_LEN       = 128
D_MODEL       = 256
N_HEADS       = 8
D_FF          = 1024
BATCH_SIZE    = 32
LR            = 3e-4
STEPS         = 30000
EVAL_EVERY    = 500
EVAL_STEPS    = 30

# Mind Palace
N_ROOMS       = 2       # start tiny — rooms earn the right to exist
MAX_ROOMS     = 16
MIN_ROOMS     = 1
HEAT_DECAY    = 0.990
DELETE_THRESH = 0.10    # was 0.25 — give rooms more time before deleting
SPLIT_THRESH  = 0.95
SPLIT_STREAK  = 5       # was 5 — wait longer before splitting again
SPAWN_PAT     = 400     # steps before first manage call
MAX_HOPS      = 3       # max rooms to traverse per forward pass

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data.txt"
TARGET    = 100_000_000

with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()

# remove weird unicode symbols (fixes huge vocab problem)
text = text.encode("ascii", "ignore").decode()
# normalize whitespace
text = text.replace("\r", "")
text = text.replace("\t", " ")

print(f"Dataset: {len(text)/1e6:.1f}MB")

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
print(f"Vocab: {VOCAB_SIZE} | Train: {len(train):,} | Val: {len(val):,}")

def get_batch(split):
    d  = train if split == 'train' else val
    ix = torch.randint(len(d) - SEQ_LEN, (BATCH_SIZE,))
    x  = torch.stack([d[i:i+SEQ_LEN]     for i in ix])
    y  = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix])
    return x, y

# ══════════════════════════════════════════════════════════════════════════════
# ROOM
# A transformer block with:
#   summary  — learned vector describing what this room knows
#   heat     — usage tracker
#   edges    — learned connections to other rooms (which rooms are related)
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

        # "the label on the door" — what this room knows
        # trained by backprop to encode the room's specialization
        self.summary = nn.Parameter(torch.randn(d_model) * 0.01)

        # confidence gate — room outputs this to say "I handled this well"
        # high confidence = don't need to visit more rooms
        self.confidence_head = nn.Linear(d_model, 1)

        self.register_buffer('heat',       torch.tensor(0.5))
        self.register_buffer('hot_streak', torch.tensor(0))
        self.register_buffer('age',        torch.tensor(0))   # evals survived

    def forward(self, x, gate_score):
        residual = x
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        out = residual + gate_score.unsqueeze(-1).unsqueeze(-1) * (x - residual)
        # confidence: mean pool → scalar per batch item
        confidence = torch.sigmoid(self.confidence_head(out.mean(dim=1)))  # (B, 1)
        return out, confidence

    def update_heat(self, gate_mean):
        self.heat = self.heat * HEAT_DECAY + (1 - HEAT_DECAY) * gate_mean
        # hot_streak is updated manually at eval time, not here

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# Reads all room summaries + input context.
# Returns ranked list of rooms to visit, not just gate scores.
# ══════════════════════════════════════════════════════════════════════════════
class MindPalaceRouter(nn.Module):
    def __init__(self, d_model, n_rooms):
        super().__init__()
        self.context_proj = nn.Linear(d_model, d_model)
        self.gate_proj    = nn.Linear(d_model, n_rooms)
        # adjacency: room i → room j connection strength
        # this is the "map" — which rooms are related
        self.adjacency    = nn.Parameter(torch.eye(n_rooms) * 2 + torch.randn(n_rooms, n_rooms) * 0.1)
        self.warp_proj    = nn.Linear(d_model, n_rooms * n_rooms)

    def forward(self, x, summaries, hard=False):
        B, n = x.shape[0], summaries.shape[0]
        ctx  = self.context_proj(x.mean(1))                                    # (B, D)
        warp = self.warp_proj(ctx).view(B, n, n) * 0.1
        adj  = torch.softmax(self.adjacency.unsqueeze(0) + warp, dim=-1)       # (B, n, n)
        raw  = (ctx.unsqueeze(1) * summaries.unsqueeze(0).expand(B,-1,-1)).sum(-1)  # (B, n)
        gate_logits = self.gate_proj(ctx) + (adj * raw.unsqueeze(1)).sum(-1)   # (B, n)
        gates = torch.sigmoid(gate_logits / 2.0)
        return (gates > 0.5).float() if hard else gates

    def get_neighbors(self, room_idx, top_k=2):
        """
        Given a room we just visited, return the top_k most connected rooms.
        This is how traversal works — after visiting room A, check its edges
        to find the most relevant next rooms.
        """
        row = torch.softmax(self.adjacency[room_idx], dim=-1)
        # exclude self
        row[room_idx] = 0
        _, neighbors = row.topk(min(top_k, len(row)-1))
        return neighbors.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# MIND PALACE
# The key new behavior:
#   - traverse rooms in order of relevance
#   - stop early if confidence is high enough (simple inputs → fewer rooms)
#   - split overloaded rooms into two specialized children
#   - delete cold rooms
# ══════════════════════════════════════════════════════════════════════════════
class MindPalace(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_rooms):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff    = d_ff
        self.dev     = device   # store device explicitly — never rely on next(parameters())
        self.rooms   = nn.ModuleList([Room(d_model, n_heads, d_ff, i) for i in range(n_rooms)])
        self.router  = MindPalaceRouter(d_model, n_rooms)

    @property
    def n_rooms(self): return len(self.rooms)

    def get_summaries(self):
        return torch.stack([r.summary for r in self.rooms])

    def forward(self, x, hard=False, training=True):
        summaries   = self.get_summaries()
        gate_scores = self.router(x, summaries, hard=hard)   # (B, n_rooms)

        # ── TRAVERSAL ──
        # sort rooms by their gate score for this input (most relevant first)
        # visit them in order, accumulate output, stop when confident enough
        mean_gates   = gate_scores.mean(dim=0)                    # (n_rooms,)
        visit_order  = mean_gates.argsort(descending=True).tolist()

        out           = x
        all_gates     = gate_scores
        total_conf    = torch.zeros(x.shape[0], 1, device=x.device)
        rooms_visited = 0

        for room_idx in visit_order[:MAX_HOPS]:
            room       = self.rooms[room_idx]
            g          = gate_scores[:, room_idx]
            out, conf  = room(out, g)
            total_conf = total_conf + conf
            rooms_visited += 1
            room.update_heat(g.mean().detach().item())

            # early exit: if average confidence is high enough, stop visiting rooms
            # simple inputs stop here, complex inputs continue
            if not training and (total_conf / rooms_visited).mean().item() > 0.85:
                break

        return out, all_gates, rooms_visited

    def split_room(self, room_idx):
        dev    = self.dev
        parent = self.rooms[room_idx]
        new_id = max(r.room_id for r in self.rooms) + 1

        child_a = Room(self.d_model, self.n_heads, self.d_ff, room_id=new_id).to(dev)
        child_b = Room(self.d_model, self.n_heads, self.d_ff, room_id=new_id + 1).to(dev)

        # both children start as copies of parent then diverge via noise
        # this is safer than weight-slicing (no dimension mismatch)
        with torch.no_grad():
            parent_params = dict(parent.named_parameters())
            for (na, pa_val), (nb, pb_val) in zip(
                child_a.named_parameters(), child_b.named_parameters()
            ):
                src   = parent_params[na].data
                scale = src.abs().mean().item() * 0.3   # noise = 30% of weight magnitude
                scale = max(scale, 0.01)                # at least 0.01
                pa_val.copy_(src + torch.randn_like(src) * scale)
                pb_val.copy_(src - torch.randn_like(src) * scale)

        # summaries diverge strongly — forces router to distinguish them
        child_a.summary.data = parent.summary.data.clone() + torch.randn_like(parent.summary) * 0.3
        child_b.summary.data = parent.summary.data.clone() - torch.randn_like(parent.summary) * 0.3
        child_a.heat.fill_(0.7)   # warm start — give them a real chance
        child_b.heat.fill_(0.7)
        child_a.hot_streak.fill_(0)
        child_b.hot_streak.fill_(0)

        return child_a, child_b

    def manage_rooms(self, current_loss, loss_history):
        log = []

        # ── tick age and hot streaks at eval time ──
        for room in self.rooms:
            room.age += 1
            if room.heat.item() > SPLIT_THRESH:
                room.hot_streak += 1
            else:
                room.hot_streak.fill_(0)

        # ── DELETE cold rooms — but only if old enough ──
        if self.n_rooms > MIN_ROOMS:
            to_delete = [i for i, r in enumerate(self.rooms)
                         if r.heat.item() < DELETE_THRESH and r.age.item() >= 6]
            for i in sorted(to_delete, reverse=True):
                log.append(f"  🗑  deleted room {self.rooms[i].room_id}  (heat={self.rooms[i].heat.item():.3f})")
                self.rooms.pop(i)
            if to_delete:
                self._rebuild_router()

        # ── SPLIT overloaded rooms ──
        if self.n_rooms < MAX_ROOMS - 1:
            for i, room in enumerate(self.rooms):
                if room.hot_streak.item() >= SPLIT_STREAK:
                    log.append(f"  ✂️  splitting room {room.room_id}  (hot_streak={room.hot_streak.item()} evals)")
                    child_a, child_b = self.split_room(i)
                    self.rooms.pop(i)
                    self.rooms.insert(i,   child_a)
                    self.rooms.insert(i+1, child_b)
                    self._rebuild_router()
                    break

        # ── SPAWN if loss stalling and no split happened ──
        elif self.n_rooms < MAX_ROOMS and len(loss_history) >= 6:
            recent   = loss_history[-3] - loss_history[-1]
            earlier  = loss_history[-6] - loss_history[-4]
            ratio    = recent / (earlier + 1e-8)
            if ratio < 0.4 and recent < 0.1:
                hottest  = max(self.rooms, key=lambda r: r.heat.item())
                new_id   = max(r.room_id for r in self.rooms) + 1
                dev      = self.dev
                new_room = Room(self.d_model, self.n_heads, self.d_ff, new_id).to(dev)
                new_room.load_state_dict(hottest.state_dict())
                with torch.no_grad():
                    for p in new_room.parameters():
                        p.add_(torch.randn_like(p) * 0.01)
                new_room.room_id = new_id
                new_room.heat.fill_(0.4)
                self.rooms.append(new_room)
                self._rebuild_router()
                log.append(f"  ✨ spawned room {new_id}  (loss stalling, ratio={ratio:.2f})")

        return log

    def _rebuild_router(self):
        self.router = MindPalaceRouter(self.d_model, self.n_rooms).to(self.dev)

# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════
class MindPalaceLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)
        self.palace    = MindPalace(D_MODEL, N_HEADS, D_FF, N_ROOMS)
        self.norm      = nn.LayerNorm(D_MODEL)
        self.head      = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids, targets=None, hard=False):
        B, T     = token_ids.shape
        x        = self.token_emb(token_ids) + self.pos_emb(torch.arange(T, device=token_ids.device))
        x, gates, n_visited = self.palace(x, hard=hard, training=targets is not None)
        logits   = self.head(self.norm(x))
        loss     = None
        if targets is not None:
            ce_loss      = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            mean_gate    = gates.mean(dim=0)
            # only apply balance loss when we have 2+ rooms (var is 0 with 1 room)
            if mean_gate.shape[0] > 1:
                balance_loss = -mean_gate.var().clamp(min=-10, max=0)
                loss = ce_loss + 0.05 * balance_loss
            else:
                loss = ce_loss
        return logits, loss, n_visited

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=300, temp=0.5, hard=False):
        self.eval()
        ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            crop        = ids[:, -SEQ_LEN:]
            logits, _,_ = self(crop, hard=hard)
            probs       = F.softmax(logits[:, -1, :] / temp, dim=-1)
            next_id     = torch.multinomial(probs, 1)
            ids         = torch.cat([ids, next_id], dim=1)
        self.train()
        return decode(ids[0].tolist())

    def room_status(self):
        print(f"\n  Rooms: {self.palace.n_rooms}")
        for r in self.palace.rooms:
            h      = r.heat.item()
            h      = 0.0 if (h != h) else h   # guard against NaN
            filled = max(0, min(20, int(h * 20)))
            bar    = '█' * filled + '░' * (20 - filled)
            streak = f"  🔥 streak={r.hot_streak.item()}" if r.hot_streak.item() > 3 else ""
            print(f"  Room {r.room_id:2d}  [{bar}]  heat={h:.3f}{streak}")
        print()

# ── TRAINING ──────────────────────────────────────────────────────────────────
model     = MindPalaceLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
n_params  = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}  |  Starting rooms: {N_ROOMS}\n")

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb   = get_batch(split)
            _, loss, _ = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

loss_history = []
start        = time.time()

for step in range(STEPS):
    xb, yb      = get_batch('train')
    _, loss, nv = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        losses  = estimate_loss()
        elapsed = (time.time() - start) / 60
        loss_history.append(losses['train'])
        print(f"step {step:5d}  |  train: {losses['train']:.4f}  val: {losses['val']:.4f}"
              f"  |  rooms: {model.palace.n_rooms}  |  {elapsed:.1f} min")
        model.room_status()

        if step >= SPAWN_PAT:
            logs = model.palace.manage_rooms(losses['train'], loss_history)
            for lg in logs: print(lg)
            if logs:
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        if step > 0 and step % 2000 == 0:
            prompts = [
                "The future of artificial intelligence",
                "In this article we discuss",
                "One of the most important discoveries",
            ]
            for p in prompts:
                print(f"\n--- {p} ---")
                print(model.generate(p, max_new_tokens=150))
            print()

print("\n=== Final output ===")
print(model.generate("The future of artificial intelligence", max_new_tokens=400))
model.room_status()
torch.save(model.state_dict(), "mindpalace_v1.pt")
print("Saved to mindpalace_v1.pt")
