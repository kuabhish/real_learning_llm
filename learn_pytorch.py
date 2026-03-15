import torch
import torch.nn as nn

print(torch.backends.mps.is_available())

# ── 1. TENSORS ────────────────────────────────────────────────────────────────
x = torch.tensor([1.0, 2.0, 3.0])   # 1D tensor (like a vector)
m = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])       # 2D tensor (like a matrix)

print(x.shape)   # torch.Size([3])
print(m.shape)   # torch.Size([2, 2])

# You'll see 3D tensors everywhere in LLMs:
# shape = (batch_size, sequence_length, embedding_dim)
t = torch.zeros(2, 5, 8)   # 2 sentences, 5 tokens each, 8-dim embedding
print(t.shape)   # torch.Size([2, 5, 8])


# ── 2. AUTOGRAD — the magic ───────────────────────────────────────────────────
x = torch.tensor(3.0, requires_grad=True)  # tell PyTorch to track this
y = x ** 2 + 2 * x                         # y = x² + 2x

y.backward()       # compute dy/dx automatically
print(x.grad)      # tensor(8.) — because dy/dx = 2x+2 = 8 at x=3


# ── 3. nn.Module — how you build any model ───────────────────────────────────
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)   # linear layer: 4 inputs → 2 outputs

    def forward(self, x):
        return self.layer(x)           # this IS the forward pass

net = TinyNet()
inp = torch.randn(3, 4)   # batch of 3 samples, each with 4 features
out = net(inp)             # calls forward() automatically
print(out.shape)           # torch.Size([3, 2])


# ── 4. TRAINING LOOP — the heartbeat of all DL ───────────────────────────────
model     = TinyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

targets = torch.randn(3, 2)   # fake targets

for step in range(5):
    optimizer.zero_grad()         # 1. clear old gradients
    preds = model(inp)            # 2. forward pass
    loss  = loss_fn(preds, targets) # 3. compute loss
    loss.backward()               # 4. backprop (compute gradients)
    optimizer.step()              # 5. update weights
    print(f"step {step}  loss: {loss.item():.4f}")


# ── 5. THE KEY OPERATIONS you'll see in transformers ─────────────────────────

# Matrix multiply — the most common op in all of deep learning
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = a @ b                  # shape: (3, 5)

# Softmax — turns raw scores into probabilities (used in attention)
scores = torch.tensor([2.0, 1.0, 0.1])
probs  = torch.softmax(scores, dim=-1)
print(probs)               # tensor([0.6590, 0.2424, 0.0986])  sums to 1

# Embedding — lookup table: integer token → dense vector
embed = nn.Embedding(num_embeddings=100, embedding_dim=8)
token_ids = torch.tensor([5, 12, 3])   # 3 tokens
vecs = embed(token_ids)
print(vecs.shape)          # torch.Size([3, 8])  each token → 8-dim vector
