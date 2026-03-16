import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os

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
SPLIT_THRESH  = 0.85
SPLIT_STREAK  = 2
SPAWN_PAT     = 400
MAX_HOPS      = 3

import tiktoken
enc        = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab  # 50257
encode     = lambda s: enc.encode(s)
decode     = lambda ids: enc.decode(ids)


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENTION GATE
# ══════════════════════════════════════════════════════════════════════════════
class AttentionGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, 1.0)

    def forward(self, x, hard=False):
        ctx  = x.mean(dim=1)
        gate = torch.sigmoid(self.gate_net(ctx))
        if hard:
            gate = (gate > 0.5).float()
        return x * gate.unsqueeze(1), gate.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  SELF ATTENTION
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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  ROOM
# ══════════════════════════════════════════════════════════════════════════════
class Room(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, room_id):
        super().__init__()
        self.room_id   = room_id
        self.d_model   = d_model
        self.attn_gate = AttentionGate(d_model)
        self.attn      = SelfAttention(d_model, n_heads)
        self.ff        = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.summary   = nn.Parameter(torch.randn(d_model) * 0.01)
        self.confidence_head = nn.Linear(d_model, 1)
        self.register_buffer('heat',       torch.tensor(0.5))
        self.register_buffer('hot_streak', torch.tensor(0))
        self.register_buffer('age',        torch.tensor(0))

    def forward(self, x, hard_gate=False):
        x_gated, gate_vals = self.attn_gate(x, hard=hard_gate)
        residual = x
        x_out    = x_gated + self.attn(self.norm1(x_gated))
        x_out    = x_out   + self.ff(self.norm2(x_out))
        gate_mean = gate_vals.mean()
        out       = residual + gate_mean * (x_out - residual)
        confidence = torch.sigmoid(self.confidence_head(out.mean(dim=1)))
        return out, confidence, gate_mean.item()

    def update_heat(self, v: float):
        self.heat = self.heat * HEAT_DECAY + (1 - HEAT_DECAY) * v


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
#  KEY FIX vs v2: _rebuild_router() now copies base_adj rows/cols from the
#  old router instead of starting fresh.  New rooms get the mean of all
#  existing rows — a neutral position, not random noise.
#  This stops the "router amnesia" that caused looping after room changes.
# ══════════════════════════════════════════════════════════════════════════════
class MindPalaceRouter(nn.Module):
    def __init__(self, d_model, n_rooms):
        super().__init__()
        self.d_model      = d_model
        self.n_rooms      = n_rooms
        self.context_proj = nn.Linear(d_model, d_model)
        self.base_adj     = nn.Parameter(
            torch.eye(n_rooms) * 2 + torch.randn(n_rooms, n_rooms) * 0.1
        )
        self.warp_proj = nn.Linear(d_model, n_rooms * n_rooms)
        nn.init.zeros_(self.warp_proj.weight)
        nn.init.zeros_(self.warp_proj.bias)
        self.gate_proj = nn.Linear(d_model, n_rooms)

    def forward(self, x, summaries):
        B, n = x.shape[0], summaries.shape[0]
        ctx  = self.context_proj(x.mean(1))
        warp = self.warp_proj(ctx).view(B, n, n) * 0.1
        adj  = torch.softmax(self.base_adj.unsqueeze(0) + warp, dim=-1)
        direct         = (ctx.unsqueeze(1) * summaries.unsqueeze(0).expand(B,-1,-1)).sum(-1)
        neighbor_boost = (adj * direct.unsqueeze(1)).sum(-1)
        logits         = self.gate_proj(ctx) + neighbor_boost
        gate_scores    = torch.sigmoid(logits / 2.0)
        return gate_scores, adj

    def get_visit_order(self, gate_scores):
        return gate_scores.mean(dim=0).argsort(descending=True).tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  MIND PALACE
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
        return torch.stack([r.summary for r in self.rooms])

    def forward(self, x, hard_gate=False, training=True):
        summaries           = self.get_summaries()
        gate_scores, adj    = self.router(x, summaries)
        visit_order         = self.router.get_visit_order(gate_scores)
        out                 = x
        total_conf          = torch.zeros(x.shape[0], 1, device=x.device)
        rooms_visited       = 0
        for room_idx in visit_order[:MAX_HOPS]:
            room = self.rooms[room_idx]
            out, conf, gm = room(out, hard_gate=hard_gate)
            total_conf    += conf
            rooms_visited += 1
            room.update_heat(gm)
            if not training and (total_conf / rooms_visited).mean().item() > 0.85:
                break
        return out, gate_scores, adj, rooms_visited

    def split_room(self, room_idx):
        parent  = self.rooms[room_idx]
        new_id  = max(r.room_id for r in self.rooms) + 1
        child_a = Room(self.d_model, self.n_heads, self.d_ff, new_id    ).to(self.dev)
        child_b = Room(self.d_model, self.n_heads, self.d_ff, new_id + 1).to(self.dev)
        with torch.no_grad():
            pp = dict(parent.named_parameters())
            for (na, pa), (nb, pb) in zip(child_a.named_parameters(), child_b.named_parameters()):
                src = pp[na].data
                sc  = max(src.abs().mean().item() * 0.3, 0.01)
                pa.copy_(src + torch.randn_like(src) * sc)
                pb.copy_(src - torch.randn_like(src) * sc)
        child_a.summary.data = parent.summary.data.clone() + torch.randn_like(parent.summary) * 0.3
        child_b.summary.data = parent.summary.data.clone() - torch.randn_like(parent.summary) * 0.3
        child_a.heat.fill_(0.7);  child_b.heat.fill_(0.7)
        child_a.hot_streak.fill_(0); child_b.hot_streak.fill_(0)
        return child_a, child_b

    def manage_rooms(self, current_loss, loss_history):
        log = []
        for room in self.rooms:
            room.age += 1
            if room.heat.item() > SPLIT_THRESH:
                room.hot_streak += 1
            else:
                room.hot_streak.fill_(0)

        if self.n_rooms > MIN_ROOMS:
            to_del = [i for i, r in enumerate(self.rooms)
                      if r.heat.item() < DELETE_THRESH and r.age.item() >= 6]
            for i in sorted(to_del, reverse=True):
                log.append(f"  🗑  deleted room {self.rooms[i].room_id}  (heat={self.rooms[i].heat.item():.3f})")
                self.rooms.pop(i)
            if to_del:
                self._rebuild_router()

        if self.n_rooms < MAX_ROOMS - 1:
            for i, room in enumerate(self.rooms):
                if room.hot_streak.item() >= SPLIT_STREAK:
                    log.append(f"  ✂️  splitting room {room.room_id}  (streak={room.hot_streak.item()})")
                    ca, cb = self.split_room(i)
                    self.rooms.pop(i)
                    self.rooms.insert(i,   ca)
                    self.rooms.insert(i+1, cb)
                    self._rebuild_router(split_idx=i)
                    break

        elif self.n_rooms < MAX_ROOMS and len(loss_history) >= 6:
            recent  = loss_history[-3] - loss_history[-1]
            earlier = loss_history[-6] - loss_history[-4]
            ratio   = recent / (earlier + 1e-8)
            if ratio < 0.4 and recent < 0.1:
                hottest = max(self.rooms, key=lambda r: r.heat.item())
                new_id  = max(r.room_id for r in self.rooms) + 1
                nr      = Room(self.d_model, self.n_heads, self.d_ff, new_id).to(self.dev)
                nr.load_state_dict(hottest.state_dict())
                with torch.no_grad():
                    for p in nr.parameters(): p.add_(torch.randn_like(p) * 0.01)
                nr.room_id = new_id
                nr.heat.fill_(0.4)
                self.rooms.append(nr)
                self._rebuild_router()
                log.append(f"  ✨ spawned room {new_id}  (ratio={ratio:.2f})")

        return log

    def _rebuild_router(self, split_idx=None):
        """
        Build a new router sized for the current room count.

        STABILITY FIX:  instead of random init, we copy the old base_adj
        into the new one.  This means the router doesn't forget which rooms
        are related — it just gains a new row/col for the new room.

        split_idx : if a room was split here, interpolate its two children
                    from the parent row (old split_idx row in old adj).
        """
        old_router = self.router
        old_n      = old_router.n_rooms
        new_n      = self.n_rooms
        new_router = MindPalaceRouter(self.d_model, new_n).to(self.dev)

        with torch.no_grad():
            old_adj = old_router.base_adj.data

            if new_n == old_n + 1 and split_idx is not None:
                # ── split case: room at split_idx → two children ──
                # build a (new_n, new_n) adj that copies old knowledge
                new_adj = torch.zeros(new_n, new_n, device=self.dev)

                # copy rows/cols that didn't change (before split point)
                new_adj[:split_idx, :split_idx]       = old_adj[:split_idx, :split_idx]
                # rows after the split shift right by 1
                if split_idx < old_n - 1:
                    new_adj[split_idx+2:, split_idx+2:] = old_adj[split_idx+1:, split_idx+1:]
                    new_adj[:split_idx,   split_idx+2:] = old_adj[:split_idx,   split_idx+1:]
                    new_adj[split_idx+2:, :split_idx]   = old_adj[split_idx+1:, :split_idx]

                # two children inherit parent row/col + small divergence
                parent_row = old_adj[split_idx]
                noise = 0.05
                new_adj[split_idx,   :] = F.pad(parent_row[:split_idx], (0, new_n - split_idx))
                new_adj[split_idx+1, :] = F.pad(parent_row[:split_idx], (0, new_n - split_idx))
                new_adj[split_idx,   split_idx]   = old_adj[split_idx, split_idx]
                new_adj[split_idx+1, split_idx+1] = old_adj[split_idx, split_idx]
                new_adj[split_idx,   split_idx+1] = 0.5
                new_adj[split_idx+1, split_idx]   = 0.5
                new_adj += torch.randn_like(new_adj) * noise
                new_router.base_adj.data.copy_(new_adj)

            elif new_n < old_n:
                # ── delete case: just crop the matrix ──
                # (deleted rooms were already popped from self.rooms)
                # safest: take top-left new_n × new_n block
                new_router.base_adj.data.copy_(old_adj[:new_n, :new_n])

            else:
                # ── spawn / other: copy what we can, leave rest random ──
                copy_n = min(old_n, new_n)
                new_router.base_adj.data[:copy_n, :copy_n] = old_adj[:copy_n, :copy_n]

            # copy context_proj and gate_proj weights (input size unchanged)
            new_router.context_proj.load_state_dict(old_router.context_proj.state_dict())

            old_w = old_router.gate_proj.weight.data
            old_b = old_router.gate_proj.bias.data
            if new_n >= old_n:
                new_router.gate_proj.weight.data[:old_n] = old_w
                new_router.gate_proj.bias.data[:old_n]   = old_b
            else:
                new_router.gate_proj.weight.data = old_w[:new_n]
                new_router.gate_proj.bias.data   = old_b[:new_n]

        self.router = new_router


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
        x    = self.token_emb(token_ids) + self.pos_emb(torch.arange(T, device=token_ids.device))
        x, gate_scores, adj, n_visited = self.palace(
            x, hard_gate=hard_gate, training=targets is not None
        )
        logits = self.head(self.norm(x))
        loss   = None
        if targets is not None:
            ce   = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            mg   = gate_scores.mean(dim=0)
            loss = ce + (0.05 * -mg.var().clamp(min=-10, max=0) if mg.shape[0] > 1 else 0)
        return logits, loss, n_visited

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=300, temp=0.8, hard_gate=False):
        self.eval()
        ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            crop       = ids[:, -SEQ_LEN:]
            logits,_,_ = self(crop, hard_gate=hard_gate)
            probs      = F.softmax(logits[:, -1, :] / temp, dim=-1)
            next_id    = torch.multinomial(probs, 1)
            ids        = torch.cat([ids, next_id], dim=1)
        self.train()
        return decode(ids[0].tolist())

    def room_status(self):
        print(f"\n  Rooms: {self.palace.n_rooms}")
        for r in self.palace.rooms:
            h      = max(0.0, min(1.0, r.heat.item()))
            bar    = '█' * int(h*20) + '░' * (20 - int(h*20))
            streak = f"  🔥{r.hot_streak.item()}" if r.hot_streak.item() > 3 else ""
            print(f"  Room {r.room_id:2d}  [{bar}]  heat={h:.3f}{streak}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING  — only runs when executed directly, not when imported
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Using device: {device}")

    DATA_PATH = "data.txt"
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    text = text.replace("\r", "").replace("\t", " ")
    print(f"Dataset: {len(text)/1e6:.1f}MB")

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

    model     = MindPalaceLLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  |  Rooms: {N_ROOMS}\n")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = [model(xb, yb)[1].item() for xb, yb in
                      (get_batch(split) for _ in range(EVAL_STEPS))]
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    loss_history, start = [], time.time()

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
            print(f"step {step:5d}  train={losses['train']:.4f}  val={losses['val']:.4f}"
                  f"  rooms={model.palace.n_rooms}  {elapsed:.1f}min")
            model.room_status()

            if step >= SPAWN_PAT:
                logs = model.palace.manage_rooms(losses['train'], loss_history)
                for lg in logs: print(lg)
                if logs:
                    old_state  = optimizer.state_dict()
                    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
                    new_params = {id(p): p for p in model.parameters()}
                    for group in old_state['param_groups']:
                        for pid in group['params']:
                            if pid in old_state['state'] and pid in new_params:
                                optimizer.state[new_params[pid]] = old_state['state'][pid]

            if step > 0 and step % 2000 == 0:
                torch.save(model.state_dict(), "mindpalace_v3.pt")
                for p in ["The future of artificial intelligence",
                          "In this article we discuss",
                          "One of the most important discoveries"]:
                    print(f"\n--- {p} ---")
                    print(model.generate(p, max_new_tokens=150))

    print("\n=== Final ===")
    print(model.generate("The future of artificial intelligence", max_new_tokens=400))
    model.room_status()
    torch.save(model.state_dict(), "mindpalace_v3.pt")
    print("Saved.")
    