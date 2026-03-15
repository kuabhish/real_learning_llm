import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEQ_LEN    = 128
D_MODEL    = 256
N_HEADS    = 8
D_FF       = 1024
BATCH_SIZE = 32
LR         = 3e-4
STEPS      = 30000
EVAL_EVERY = 500
EVAL_STEPS = 30

# Mind Palace
N_ROOMS       = 2
MAX_ROOMS     = 16
MIN_ROOMS     = 1
HEAT_DECAY    = 0.990
DELETE_THRESH = 0.10
SPLIT_THRESH  = 0.9
SPLIT_STREAK  = 5
SPAWN_PAT     = 400
MAX_HOPS      = 3

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data.txt"
with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()
text = text.encode("ascii", "ignore").decode()
text = text.replace("\r", "").replace("\t", " ")
print(f"Dataset: {len(text)/1e6:.1f}MB")

chars      = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for c, i in stoi.items()}
encode     = lambda s: [stoi[c] for c in s if c in stoi]
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
#  GATE NEURON
#  Sits in front of every attention layer inside a room.
#  Learns a scalar decision per token per head:
#    - near 0  → block this attention path (skip it)
#    - near 1  → pass through normally
#    - >1      → amplify (the room is "excited" by this input)
#  This is the "neuron in front of attention" you asked for.
#  At inference it can hard-threshold to 0/1 for speed.
# ══════════════════════════════════════════════════════════════════════════════
class AttentionGate(nn.Module):
    """
    Learned gate that sits *before* self-attention.
    Input  : x  (B, T, D)
    Output : gated x  (B, T, D)  +  gate scalar for logging

    The gate projects the mean-pooled context to a single scalar,
    then multiplies the entire sequence. Simple, differentiable, fast.
    The gate is per-room — each room learns independently whether
    it should fire on a given input.
    """
    def __init__(self, d_model):
        super().__init__()
        # two-layer MLP: context → hidden → scalar
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        # bias toward open (sigmoid(1) ≈ 0.73) so rooms start active
        nn.init.constant_(self.gate_net[-1].bias, 1.0)

    def forward(self, x, hard=False):
        ctx   = x.mean(dim=1)                   # (B, D) — mean pool over time
        raw   = self.gate_net(ctx)               # (B, 1)
        gate  = torch.sigmoid(raw)               # (B, 1) in [0, 1]
        if hard:
            gate = (gate > 0.5).float()          # binary at inference if requested
        # broadcast over T and D
        return x * gate.unsqueeze(1), gate.squeeze(-1)   # (B,T,D), (B,)


# ══════════════════════════════════════════════════════════════════════════════
#  SELF ATTENTION  (unchanged from v1)
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
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        s = s.masked_fill(mask, float('-inf'))
        return self.W_o((F.softmax(s, -1) @ V).transpose(1,2).contiguous().view(B,T,C))


# ══════════════════════════════════════════════════════════════════════════════
#  ROOM  (v2)
#  Changes vs v1:
#    - AttentionGate sits before self-attention (your "neuron in front")
#    - gate_value stored for heat updates (replaces old gate_score arg)
#    - summary now has a *projection head* so the router can read it cleanly
#    - confidence head kept — drives early exit
# ══════════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)


class Room(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, room_id):
        super().__init__()
        self.room_id = room_id
        self.d_model = d_model

        # ── THE KEY NEW PIECE: gate neuron before attention ──
        self.attn_gate  = AttentionGate(d_model)

        self.attn  = SelfAttention(d_model, n_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # "what this room knows" — trained by backprop
        # think of it as the label on the door
        self.summary = nn.Parameter(torch.randn(d_model) * 0.01)

        # confidence: room says "I handled this — no need to go further"
        self.confidence_head = nn.Linear(d_model, 1)

        self.register_buffer('heat',       torch.tensor(0.5))
        self.register_buffer('hot_streak', torch.tensor(0))
        self.register_buffer('age',        torch.tensor(0))

    def forward(self, x, hard_gate=False):
        """
        x          : (B, T, D)
        hard_gate  : if True, gate is binary (0 or 1) — faster inference
        returns    : out (B,T,D), confidence (B,1), gate_mean scalar
        """
        # 1. Gate neuron decides how much attention fires for this input
        x_gated, gate_vals = self.attn_gate(x, hard=hard_gate)  # gate_vals: (B,)

        # 2. Normal transformer block on the gated input
        residual = x
        normed   = self.norm1(x_gated)
        x_out    = x_gated + self.attn(normed)
        x_out    = x_out + self.ff(self.norm2(x_out))

        # 3. Blend with residual scaled by gate mean
        #    If gate=0, output = input (room did nothing)
        #    If gate=1, output = full transformer output
        gate_mean = gate_vals.mean()
        out       = residual + gate_mean * (x_out - residual)

        # 4. Confidence: how sure is this room it handled the input?
        confidence = torch.sigmoid(self.confidence_head(out.mean(dim=1)))  # (B,1)

        return out, confidence, gate_mean.item()

    def update_heat(self, gate_mean_val: float):
        self.heat = self.heat * HEAT_DECAY + (1 - HEAT_DECAY) * gate_mean_val


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER  (v2)
#  New: dynamic adjacency warping
#
#  Fixed adjacency (v1): room distances are constant during inference
#  Dynamic adjacency (v2): input context *warps* the room graph
#    → "Paris" input pulls geography rooms closer together
#    → "def foo" input pulls syntax rooms closer together
#
#  How it works:
#    base_adj   : learned N×N matrix (who is normally close to whom)
#    warp_proj  : projects input context to a N×N delta matrix
#    final_adj  = softmax(base_adj + warp_delta)
#    This means the *same rooms* but at *different distances* per input.
#
#  The gate scores then use both direct similarity (ctx · summary)
#  AND neighborhood influence (adj × scores) — so activating room A
#  naturally pulls in its neighbors.
# ══════════════════════════════════════════════════════════════════════════════
class MindPalaceRouter(nn.Module):
    def __init__(self, d_model, n_rooms):
        super().__init__()
        self.d_model      = d_model
        self.n_rooms      = n_rooms
        self.context_proj = nn.Linear(d_model, d_model)

        # base adjacency: who sits next to whom in the palace
        self.base_adj = nn.Parameter(
            torch.eye(n_rooms) * 2 + torch.randn(n_rooms, n_rooms) * 0.1
        )

        # warp projection: context → N×N delta on the adjacency map
        # this is the v2 dynamic topology piece
        self.warp_proj = nn.Linear(d_model, n_rooms * n_rooms)
        nn.init.zeros_(self.warp_proj.weight)   # start with no warping
        nn.init.zeros_(self.warp_proj.bias)

        # final gate: context → room scores
        self.gate_proj = nn.Linear(d_model, n_rooms)

    def forward(self, x, summaries):
        """
        x         : (B, T, D)
        summaries : (n_rooms, D)  — one vector per room describing what it knows
        returns   : gate_scores (B, n_rooms) in [0,1]
                    adj         (B, n_rooms, n_rooms) — warped adjacency for logging
        """
        B, n = x.shape[0], summaries.shape[0]

        # context vector from input
        ctx = self.context_proj(x.mean(1))    # (B, D)

        # ── dynamic adjacency ──
        warp  = self.warp_proj(ctx).view(B, n, n) * 0.1    # (B, n, n)
        adj   = torch.softmax(
            self.base_adj.unsqueeze(0) + warp, dim=-1
        )                                                   # (B, n, n)

        # ── direct similarity: how much does ctx match each room's summary? ──
        direct = (ctx.unsqueeze(1) * summaries.unsqueeze(0).expand(B,-1,-1)).sum(-1)  # (B,n)

        # ── neighborhood boost: activated rooms pull in neighbors ──
        neighbor_boost = (adj * direct.unsqueeze(1)).sum(-1)   # (B,n)

        # ── final gate scores ──
        logits      = self.gate_proj(ctx) + neighbor_boost     # (B,n)
        gate_scores = torch.sigmoid(logits / 2.0)              # (B,n) in [0,1]

        return gate_scores, adj

    def get_visit_order(self, gate_scores):
        """
        Returns room indices sorted by relevance (most relevant first).
        Used by the palace to decide traversal order.
        """
        mean_gates  = gate_scores.mean(dim=0)    # (n_rooms,)
        visit_order = mean_gates.argsort(descending=True).tolist()
        return visit_order


# ══════════════════════════════════════════════════════════════════════════════
#  MIND PALACE  (v2)
#  Traversal logic:
#    1. Router scores all rooms (who is relevant to this input?)
#    2. Sort by score — most relevant first
#    3. Visit up to MAX_HOPS rooms
#    4. After each room, check confidence — if high enough, stop early
#    5. Each room's AttentionGate fires independently (soft or hard)
#
#  Room management (training only):
#    - DELETE  : cold rooms (heat < thresh) after age >= 6 evals
#    - SPLIT   : overloaded rooms (hot_streak >= SPLIT_STREAK)
#    - SPAWN   : if loss is stalling and no split happened
# ══════════════════════════════════════════════════════════════════════════════
class MindPalace(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_rooms):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff    = d_ff
        self.dev     = device
        self.rooms   = nn.ModuleList(
            [Room(d_model, n_heads, d_ff, i) for i in range(n_rooms)]
        )
        self.router  = MindPalaceRouter(d_model, n_rooms)

    @property
    def n_rooms(self): return len(self.rooms)

    def get_summaries(self):
        return torch.stack([r.summary for r in self.rooms])   # (n_rooms, D)

    def forward(self, x, hard_gate=False, training=True):
        summaries              = self.get_summaries()
        gate_scores, adj       = self.router(x, summaries)    # (B,n), (B,n,n)
        visit_order            = self.router.get_visit_order(gate_scores)

        out           = x
        total_conf    = torch.zeros(x.shape[0], 1, device=x.device)
        rooms_visited = 0

        for room_idx in visit_order[:MAX_HOPS]:
            room = self.rooms[room_idx]
            out, conf, gate_mean_val = room(out, hard_gate=hard_gate)
            total_conf    += conf
            rooms_visited += 1
            room.update_heat(gate_mean_val)

            # early exit at inference: simple inputs stop after 1–2 rooms
            if not training and (total_conf / rooms_visited).mean().item() > 0.85:
                break

        return out, gate_scores, adj, rooms_visited

    # ── SPLIT ────────────────────────────────────────────────────────────────
    def split_room(self, room_idx):
        parent = self.rooms[room_idx]
        new_id = max(r.room_id for r in self.rooms) + 1
        child_a = Room(self.d_model, self.n_heads, self.d_ff, new_id    ).to(self.dev)
        child_b = Room(self.d_model, self.n_heads, self.d_ff, new_id + 1).to(self.dev)

        with torch.no_grad():
            parent_params = dict(parent.named_parameters())
            for (na, pa_val), (nb, pb_val) in zip(
                child_a.named_parameters(), child_b.named_parameters()
            ):
                src   = parent_params[na].data
                scale = max(src.abs().mean().item() * 0.3, 0.01)
                pa_val.copy_(src + torch.randn_like(src) * scale)
                pb_val.copy_(src - torch.randn_like(src) * scale)

        # summaries diverge — forces router to tell them apart
        child_a.summary.data = parent.summary.data.clone() + torch.randn_like(parent.summary) * 0.3
        child_b.summary.data = parent.summary.data.clone() - torch.randn_like(parent.summary) * 0.3
        child_a.heat.fill_(0.7)
        child_b.heat.fill_(0.7)
        child_a.hot_streak.fill_(0)
        child_b.hot_streak.fill_(0)
        return child_a, child_b

    # ── MANAGE ───────────────────────────────────────────────────────────────
    def manage_rooms(self, current_loss, loss_history):
        log = []

        for room in self.rooms:
            room.age += 1
            if room.heat.item() > SPLIT_THRESH:
                room.hot_streak += 1
            else:
                room.hot_streak.fill_(0)

        # DELETE cold rooms
        if self.n_rooms > MIN_ROOMS:
            to_delete = [i for i, r in enumerate(self.rooms)
                         if r.heat.item() < DELETE_THRESH and r.age.item() >= 6]
            for i in sorted(to_delete, reverse=True):
                log.append(f"  🗑  deleted room {self.rooms[i].room_id}  (heat={self.rooms[i].heat.item():.3f})")
                self.rooms.pop(i)
            if to_delete:
                self._rebuild_router()

        # SPLIT overloaded rooms
        if self.n_rooms < MAX_ROOMS - 1:
            for i, room in enumerate(self.rooms):
                if room.hot_streak.item() >= SPLIT_STREAK:
                    log.append(f"  ✂️  splitting room {room.room_id}  (streak={room.hot_streak.item()})")
                    child_a, child_b = self.split_room(i)
                    self.rooms.pop(i)
                    self.rooms.insert(i,   child_a)
                    self.rooms.insert(i+1, child_b)
                    self._rebuild_router()
                    break

        # SPAWN on stall
        elif self.n_rooms < MAX_ROOMS and len(loss_history) >= 6:
            recent  = loss_history[-3] - loss_history[-1]
            earlier = loss_history[-6] - loss_history[-4]
            ratio   = recent / (earlier + 1e-8)
            if ratio < 0.4 and recent < 0.1:
                hottest  = max(self.rooms, key=lambda r: r.heat.item())
                new_id   = max(r.room_id for r in self.rooms) + 1
                new_room = Room(self.d_model, self.n_heads, self.d_ff, new_id).to(self.dev)
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
#  FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════
class MindPalaceLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)
        self.palace    = MindPalace(D_MODEL, N_HEADS, D_FF, N_ROOMS)
        self.norm      = nn.LayerNorm(D_MODEL)
        self.head      = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids, targets=None, hard_gate=False):
        B, T = token_ids.shape
        pos  = torch.arange(T, device=token_ids.device)
        x    = self.token_emb(token_ids) + self.pos_emb(pos)

        x, gate_scores, adj, n_visited = self.palace(
            x, hard_gate=hard_gate, training=targets is not None
        )
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

            # balance loss: encourage gates to vary (avoid all-same scores)
            mean_gate = gate_scores.mean(dim=0)
            if mean_gate.shape[0] > 1:
                balance_loss = -mean_gate.var().clamp(min=-10, max=0)
                loss = ce_loss + 0.05 * balance_loss
            else:
                loss = ce_loss

        return logits, loss, n_visited

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=300, temp=0.5, hard_gate=False):
        self.eval()
        ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            crop        = ids[:, -SEQ_LEN:]
            logits,_,_  = self(crop, hard_gate=hard_gate)
            probs       = F.softmax(logits[:, -1, :] / temp, dim=-1)
            next_id     = torch.multinomial(probs, 1)
            ids         = torch.cat([ids, next_id], dim=1)
        self.train()
        return decode(ids[0].tolist())

    def room_status(self):
        print(f"\n  Rooms: {self.palace.n_rooms}")
        for r in self.palace.rooms:
            h      = max(0.0, min(1.0, r.heat.item()))
            filled = int(h * 20)
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
            xb, yb     = get_batch(split)
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
torch.save(model.state_dict(), "mindpalace_v2.pt")
print("Saved to mindpalace_v2.pt")
