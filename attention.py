import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── CONFIG ────────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 256    # number of unique tokens
SEQ_LEN     = 16     # how many tokens we process at once
D_MODEL     = 64     # size of each token's embedding vector
N_HEADS     = 4      # number of attention heads
N_LAYERS    = 2      # how many transformer blocks to stack
D_FF        = 256    # size of the feedforward hidden layer


# ── PIECE 1: SELF-ATTENTION ───────────────────────────────────────────────────
# This is THE core idea. Every token looks at every other token
# and decides how much to attend to it.

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads  # each head works in smaller space

        # Three linear projections: Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output projection

    def forward(self, x):
        B, T, C = x.shape   # (batch, seq_len, d_model)

        # Project input into Q, K, V
        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        # Reshape: (B, T, C) → (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores: how much does each token attend to each other?
        scale  = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale  # (B, n_heads, T, T)

        # Causal mask: token i can only see tokens 0..i (not the future)
        mask   = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax → probabilities, then weighted sum of values
        attn   = F.softmax(scores, dim=-1)   # (B, n_heads, T, T)
        out    = attn @ V                     # (B, n_heads, T, d_head)

        # Merge heads back together
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# ── PIECE 2: FEEDFORWARD NETWORK ─────────────────────────────────────────────
# After attention mixes information across tokens,
# this transforms each token independently.
# Think of it as the model's "memory" or "knowledge store".

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # smooth activation (better than ReLU for LLMs)
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ── PIECE 3: TRANSFORMER BLOCK ───────────────────────────────────────────────
# One full block = attention + feedforward, each wrapped with
# residual connection ("add") and layer normalization ("norm").
# Residual connections let gradients flow easily during training.

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn  = SelfAttention(d_model, n_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # attention  + residual
        x = x + self.ff(self.norm2(x))    # feedforward + residual
        return x


# ── PIECE 4: FULL LANGUAGE MODEL ─────────────────────────────────────────────
# Stack N blocks, add an embedding layer at the front
# and a prediction head at the end.

class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)   # learned position info
        self.blocks    = nn.Sequential(*[
            TransformerBlock(D_MODEL, N_HEADS, D_FF)
            for _ in range(N_LAYERS)
        ])
        self.norm      = nn.LayerNorm(D_MODEL)
        self.head      = nn.Linear(D_MODEL, VOCAB_SIZE)    # → logits over vocab

    def forward(self, token_ids):
        B, T = token_ids.shape

        tok = self.token_emb(token_ids)                  # (B, T, D_MODEL)
        print(tok.shape)
        pos = self.pos_emb(torch.arange(T))              # (T,  D_MODEL)
        print(pos.shape)
        x   = tok + pos                                  # add position info
        print(x.shape)
        x   = self.blocks(x)
        print(x.shape)
        x   = self.norm(x)
        print(x.shape)
        logits = self.head(x)                            # (B, T, VOCAB_SIZE)
        return logits


# ── QUICK TEST ────────────────────────────────────────────────────────────────
model = TinyLLM()

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")   # ~200k — tiny but real

# Fake input: batch of 2 sequences, each 16 tokens
x      = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
logits = model(x)
print(f"Input shape:  {x.shape}")       # (2, 16)
print(f"Output shape: {logits.shape}")  # (2, 16, 256) — a prediction for each position
