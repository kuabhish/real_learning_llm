import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── DEVICE — use Mac GPU if available ─────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 65     # we'll build a char-level vocab from the text
SEQ_LEN     = 64     # characters per training chunk
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 3
D_FF        = 512
BATCH_SIZE  = 32
LR          = 3e-4
STEPS       = 3000
EVAL_EVERY  = 300

# ── DATA — tiny Shakespeare (built right in, no download needed) ──────────────
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep,
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action.
All the world's a stage, and all the men and women merely players.
They have their exits and their entrances, and one man in his time plays many parts.
Friends, Romans, countrymen, lend me your ears.
I come to bury Caesar, not to praise him.
The evil that men do lives after them; the good is oft interred with their bones.
What a piece of work is a man! How noble in reason, how infinite in faculty!
We are such stuff as dreams are made on, and our little life is rounded with a sleep.
""" * 20   # repeat 20x so we have enough training data (~30k chars)

# ── TOKENIZER — character level (simplest possible) ───────────────────────────
# Each unique character gets an integer ID
chars     = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi      = {c: i for i, c in enumerate(chars)}   # char → int
itos      = {i: c for c, i in stoi.items()}        # int  → char

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

data   = torch.tensor(encode(text), dtype=torch.long).to(device)
n      = int(0.9 * len(data))
train  = data[:n]
val    = data[n:]

print(f"Vocab size: {VOCAB_SIZE} characters")
print(f"Training tokens: {len(train):,}")

# ── BATCH SAMPLER — grab random chunks of text ────────────────────────────────
def get_batch(split):
    d      = train if split == 'train' else val
    ix     = torch.randint(len(d) - SEQ_LEN, (BATCH_SIZE,))
    x      = torch.stack([d[i:i+SEQ_LEN]   for i in ix])     # input
    y      = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix])   # target = shifted by 1
    return x, y

# ── MODEL ─────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_q     = nn.Linear(d_model, d_model)
        self.W_k     = nn.Linear(d_model, d_model)
        self.W_v     = nn.Linear(d_model, d_model)
        self.W_o     = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        out    = attn @ V
        out    = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn  = SelfAttention(d_model, n_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks    = nn.Sequential(*[
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
        logits = self.head(x)              # (B, T, VOCAB_SIZE)

        loss = None
        if targets is not None:
            # reshape for cross entropy: (B*T, VOCAB_SIZE) vs (B*T,)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=200):
        self.eval()
        ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            ids_crop  = ids[:, -SEQ_LEN:]             # keep last SEQ_LEN tokens
            logits, _ = self(ids_crop)
            probs     = F.softmax(logits[:, -1, :], dim=-1)  # last token's probs
            next_id   = torch.multinomial(probs, num_samples=1)
            ids       = torch.cat([ids, next_id], dim=1)
        return decode(ids[0].tolist())

# ── TRAINING ──────────────────────────────────────────────────────────────────
model     = TinyLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
n_params  = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}\n")

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        total = 0
        for _ in range(20):
            xb, yb  = get_batch(split)
            _, loss = model(xb, yb)
            total  += loss.item()
        losses[split] = total / 20
    model.train()
    return losses

print("Starting training...\n")
for step in range(STEPS):
    xb, yb    = get_batch('train')
    _, loss   = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        losses = estimate_loss()
        print(f"step {step:4d}  |  train loss: {losses['train']:.4f}  |  val loss: {losses['val']:.4f}")

# ── GENERATE — see what it learned ────────────────────────────────────────────
print("\n--- Generated text after training ---\n")
print(model.generate("To be", max_new_tokens=300))
