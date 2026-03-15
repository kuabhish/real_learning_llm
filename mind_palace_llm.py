import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, urllib.request, time

# ── DEVICE ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEQ_LEN        = 128
D_MODEL        = 256
N_HEADS        = 8
D_FF           = 1024
BATCH_SIZE     = 32
LR             = 3e-4
STEPS          = 10000
EVAL_EVERY     = 500
EVAL_STEPS     = 30

# Mind Palace specific
N_ROOMS        = 1      # START WITH 1 — rooms are earned, not given
MAX_ROOMS      = 10     # never grow beyond this
MIN_ROOMS      = 1      # never shrink below this
HEAT_DECAY     = 0.990  # faster decay — unused rooms cool in ~300 steps
DELETE_THRESH  = 0.30   # delete if heat drops below 30% — actually reachable now
SPAWN_PATIENCE = 300    # wait this many steps before first spawn check

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data.txt"
if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
with open(DATA_PATH) as f:
    text = f.read()

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
# PIECE 1 — A SINGLE ROOM
# Each room is a standard transformer block PLUS a summary vector.
# The summary vector is a learned D_MODEL-dim description of what this room knows.
# The router reads these summaries to decide which rooms to activate.
# ══════════════════════════════════════════════════════════════════════════════

class Room(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, room_id):
        super().__init__()
        self.room_id = room_id

        # Standard transformer block (attention + feedforward)
        self.attn  = SelfAttention(d_model, n_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ── THE NEW PART ──
        # Summary vector: a single learned vector that describes what this room
        # specializes in. Starts random, gets updated by backprop just like weights.
        # Think of it as the "label on the door" of this room.
        self.summary = nn.Parameter(torch.randn(d_model) * 0.01)

        # Heat: tracks how often this room is activated (not a nn.Parameter,
        # just a float we update manually — gradients don't flow through it)
        self.register_buffer('heat', torch.tensor(1.0))

    def forward(self, x, gate_score):
        # gate_score: scalar 0..1 — how strongly to apply this room's output
        # If gate_score ≈ 0, the room contributes almost nothing (effectively skipped)
        # If gate_score ≈ 1, the room contributes fully
        residual = x
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        # Blend: gate_score=1 means full room output, gate_score=0 means pass through unchanged
        return residual + gate_score.unsqueeze(-1).unsqueeze(-1) * (x - residual)

    def update_heat(self, gate_score_mean):
        # Decay heat slowly every step, then add the mean gate score for this step
        # Rooms that get activated a lot stay hot. Rooms that don't get activated cool down.
        self.heat = self.heat * HEAT_DECAY + (1 - HEAT_DECAY) * gate_score_mean


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 2 — SELF ATTENTION (unchanged from before)
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
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 3 — THE ROUTER  (v1 + v2 combined)
#
# v1: reads room summaries → computes relevance scores → gates each room
# v2: computes a context vector from input → warps the adjacency between rooms
#     so "nearby" rooms shift based on what's being processed
# ══════════════════════════════════════════════════════════════════════════════

class MindPalaceRouter(nn.Module):
    def __init__(self, d_model, n_rooms):
        super().__init__()
        self.d_model = d_model

        # Compresses the input sequence into a single context vector
        # (mean pool → linear → the "what am I thinking about?" vector)
        self.context_proj = nn.Linear(d_model, d_model)

        # Gate projection: context vector → one score per room
        self.gate_proj = nn.Linear(d_model, n_rooms)

        # ── V2: DYNAMIC TOPOLOGY ──
        # Adjacency matrix: learned base distances between rooms
        # Shape (n_rooms, n_rooms) — how related is room i to room j?
        # This is the "map" of the mind palace.
        self.adjacency = nn.Parameter(torch.eye(n_rooms) + torch.randn(n_rooms, n_rooms) * 0.1)

        # Context warp: given the context vector, output a small delta to the adjacency
        # This is v2 — the map warps based on what we're thinking about
        self.warp_proj  = nn.Linear(d_model, n_rooms * n_rooms)

    def forward(self, x, room_summaries, inference_mode=False):
        # x: (B, T, D_MODEL)
        # room_summaries: (n_rooms, D_MODEL) — the "door labels"

        B = x.shape[0]
        n_rooms = room_summaries.shape[0]

        # Step 1 — compress input sequence to a single context vector
        # Mean pool across the sequence dimension → (B, D_MODEL)
        context = self.context_proj(x.mean(dim=1))   # (B, D_MODEL)

        # Step 2 — v2: compute context-dependent warp of the adjacency matrix
        # warp_proj outputs (B, n_rooms * n_rooms), reshape to (B, n_rooms, n_rooms)
        warp   = self.warp_proj(context).view(B, n_rooms, n_rooms) * 0.1
        # Add warp to base adjacency (same base map, shifted by context)
        adj    = self.adjacency.unsqueeze(0) + warp   # (B, n_rooms, n_rooms)
        adj    = torch.softmax(adj, dim=-1)            # normalize rows

        # Step 3 — score each room by matching context against room summaries
        # room_summaries: (n_rooms, D) → (B, n_rooms, D) after expand
        summaries_exp = room_summaries.unsqueeze(0).expand(B, -1, -1)  # (B, n_rooms, D)
        # dot product of context with each summary
        context_exp   = context.unsqueeze(1)                            # (B, 1, D)
        raw_scores    = (context_exp * summaries_exp).sum(-1)           # (B, n_rooms)

        # Step 4 — apply adjacency warp to scores
        # Each room's final score is influenced by its neighbors' raw scores
        # This makes the routing "topology-aware" — nearby rooms pull each other up
        warped_scores = (adj * raw_scores.unsqueeze(1)).sum(-1)         # (B, n_rooms)

        # Step 5 — sigmoid gives independent gate per room (not softmax!)
        # Softmax forces rooms to compete. Sigmoid lets multiple rooms activate together.
        # This is important — a complex input should activate several rooms at once.
        gate_scores = torch.sigmoid(warped_scores)   # (B, n_rooms) each in [0, 1]

        # At inference: optionally hard-threshold — rooms either fully ON or OFF
        # This makes inference faster (skip rooms with gate < 0.5 entirely)
        if inference_mode:
            gate_scores = (gate_scores > 0.5).float()

        return gate_scores   # (B, n_rooms)


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 4 — THE MIND PALACE
# Holds all the rooms + the router. Handles dynamic create/delete.
# This is the main thing that replaces nn.Sequential in the old model.
# ══════════════════════════════════════════════════════════════════════════════

class MindPalace(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_rooms):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff    = d_ff
        # nn.ModuleList so PyTorch tracks all room parameters properly
        self.rooms   = nn.ModuleList([
            Room(d_model, n_heads, d_ff, room_id=i) for i in range(n_rooms)
        ])
        self.router  = MindPalaceRouter(d_model, n_rooms)
        self._step   = 0   # internal step counter for spawn decisions

    @property
    def n_rooms(self):
        return len(self.rooms)

    def get_summaries(self):
        # Stack all room summary vectors into one matrix (n_rooms, D_MODEL)
        return torch.stack([r.summary for r in self.rooms])

    def forward(self, x):
        summaries   = self.get_summaries()               # (n_rooms, D)
        gate_scores = self.router(x, summaries)          # (B, n_rooms)

        # Pass x through every room, weighted by its gate score
        out = x
        for i, room in enumerate(self.rooms):
            g    = gate_scores[:, i]       # (B,) — gate score for this room
            out  = room(out, g)
            # Update heat (detach so this doesn't affect gradients)
            room.update_heat(g.mean().detach().item())

        return out

    def manage_rooms(self, current_loss, prev_loss, loss_history):
        """
        Called every N steps during training.
        Deletes cold rooms and spawns new ones if needed.
        Returns a log string describing what happened.

        Spawn logic — smarter than before:
          We keep a window of recent losses. If the improvement rate is
          slowing down (we're on a plateau), spawn a new room to give the
          model more capacity to break through.
        """
        log = []

        # ── DELETE cold rooms ──
        if self.n_rooms > MIN_ROOMS:
            to_delete = [i for i, r in enumerate(self.rooms)
                         if r.heat.item() < DELETE_THRESH]
            for i in sorted(to_delete, reverse=True):
                log.append(f"  🗑  deleted room {self.rooms[i].room_id} "
                           f"(heat={self.rooms[i].heat.item():.3f})")
                self.rooms.pop(i)
            if to_delete:
                self._rebuild_router()

        # ── SPAWN when improvement is slowing ──
        if self.n_rooms < MAX_ROOMS and len(loss_history) >= 3:
            # How much did loss improve over the last 3 evals?
            recent_drop   = loss_history[-3] - loss_history[-1]
            # How much did it improve in the 3 evals before that?
            if len(loss_history) >= 6:
                earlier_drop = loss_history[-6] - loss_history[-4]
            else:
                earlier_drop = recent_drop + 1   # force no spawn early on

            # Spawn if recent improvement is less than 40% of earlier improvement
            # i.e. we're clearly slowing down and more capacity might help
            improvement_ratio = recent_drop / (earlier_drop + 1e-8)
            if improvement_ratio < 0.4 and recent_drop < 0.1:
                new_id   = max(r.room_id for r in self.rooms) + 1
                new_room = Room(self.d_model, self.n_heads, self.d_ff,
                                room_id=new_id).to(next(self.parameters()).device)
                new_room.heat.fill_(0.5)   # warm start — give it a fair chance
                self.rooms.append(new_room)
                self._rebuild_router()
                log.append(f"  ✨ spawned room {new_id} "
                           f"(improvement slowing: ratio={improvement_ratio:.2f}, "
                           f"loss={current_loss:.4f})")

        return log

    def _rebuild_router(self):
        """Rebuild router when number of rooms changes."""
        dev = next(self.parameters()).device
        self.router = MindPalaceRouter(self.d_model, self.n_rooms).to(dev)


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 5 — FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MindPalaceLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb    = nn.Embedding(SEQ_LEN, D_MODEL)
        self.palace     = MindPalace(D_MODEL, N_HEADS, D_FF, N_ROOMS)
        self.norm       = nn.LayerNorm(D_MODEL)
        self.head       = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids, targets=None):
        B, T   = token_ids.shape
        x      = self.token_emb(token_ids) + self.pos_emb(torch.arange(T, device=token_ids.device))
        x      = self.palace(x)
        x      = self.norm(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=300, temperature=0.8, inference_mode=False):
        self.eval()
        # Temporarily patch router to use inference_mode if requested
        original_forward = self.palace.router.forward
        if inference_mode:
            def inf_forward(x, summaries):
                return original_forward(x, summaries, inference_mode=True)
            self.palace.router.forward = inf_forward

        ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            crop      = ids[:, -SEQ_LEN:]
            logits, _ = self(crop)
            probs     = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id   = torch.multinomial(probs, 1)
            ids       = torch.cat([ids, next_id], dim=1)

        if inference_mode:
            self.palace.router.forward = original_forward
        self.train()
        return decode(ids[0].tolist())

    def room_status(self):
        print(f"\n  Rooms: {self.palace.n_rooms}")
        for r in self.palace.rooms:
            bar  = '█' * int(r.heat.item() * 20)
            print(f"  Room {r.room_id:2d}  heat={r.heat.item():.3f}  {bar}")
        print()


# ── TRAINING ──────────────────────────────────────────────────────────────────
model     = MindPalaceLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
n_params  = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")
print(f"Starting with {N_ROOMS} rooms\n")

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = [model(*get_batch(split))[1].item() for _ in range(EVAL_STEPS)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

loss_history = []   # track eval losses over time for spawn decisions
start        = time.time()

for step in range(STEPS):
    xb, yb  = get_batch('train')
    _, loss = model(xb, yb)

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

        # Show room heat map
        model.room_status()

        # Let the palace manage itself — delete cold rooms, spawn if needed
        if step >= SPAWN_PATIENCE:
            prev = loss_history[-2] if len(loss_history) >= 2 else float('inf')
            logs = model.palace.manage_rooms(losses['train'], prev, loss_history)
            for lg in logs:
                print(lg)
            if logs:
                # Rebuild optimizer whenever rooms change (parameters added/removed)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        if step > 0 and step % 2000 == 0:
            print("\n--- sample (normal) ---")
            print(model.generate("ROMEO:", max_new_tokens=200))
            print("\n--- sample (inference mode: hard gates) ---")
            print(model.generate("ROMEO:", max_new_tokens=200, inference_mode=True))
            print()

print("\n=== Final output ===")
print(model.generate("ROMEO:", max_new_tokens=400))
model.room_status()
torch.save(model.state_dict(), "mindpalace_v1.pt")
print("Saved to mindpalace_v1.pt")
