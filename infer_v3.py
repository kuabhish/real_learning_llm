import torch
import torch.nn.functional as F
import tiktoken

# ── point these at your files ─────────────────────────────────────────────────
CHECKPOINT  = "mindpalace_v3.pt"
TRAIN_FILE  = "mindpalacev3.py"          # your original training script
SEQ_LEN     = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── load model class from training file, then load weights ───────────────────
import importlib.util, sys

spec = importlib.util.spec_from_file_location("train", TRAIN_FILE)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)          # runs the file — loads data too
                                            # (that's fine, it's fast)

model = train_mod.MindPalaceLLM().to(device)

# auto-detect room count from checkpoint and resize if needed
state   = torch.load(CHECKPOINT, map_location=device)
n_rooms = max(int(k.split(".")[2]) for k in state if k.startswith("palace.rooms.")) + 1

if n_rooms != train_mod.N_ROOMS:
    print(f"Checkpoint has {n_rooms} rooms (config says {train_mod.N_ROOMS}). Rebuilding…")
    from torch import nn
    model.palace.rooms = nn.ModuleList([
        train_mod.Room(train_mod.D_MODEL, train_mod.N_HEADS, train_mod.D_FF, i)
        for i in range(n_rooms)
    ]).to(device)
    model.palace.router = train_mod.MindPalaceRouter(train_mod.D_MODEL, n_rooms).to(device)

model.load_state_dict(state)
model.eval()
print(f"Loaded  '{CHECKPOINT}'  — {n_rooms} rooms  "
      f"{sum(p.numel() for p in model.parameters()):,} params\n")

enc    = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda ids: enc.decode(ids)

# ── generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_tokens=200, temp=0.8, top_p=0.9) -> str:
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        crop   = ids[:, -SEQ_LEN:]
        logits = model(crop)[0][:, -1, :] / temp

        # nucleus (top-p) sampling
        probs  = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        sorted_probs[cumsum - sorted_probs > top_p] = 0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]

        ids = torch.cat([ids, next_id.view(1,1)], dim=1)
        if next_id.item() == enc.eot_token:
            break

    return decode(ids[0].tolist())[len(prompt):]   # strip the prompt

# ── chat loop ─────────────────────────────────────────────────────────────────
cfg = {"temp": 0.8, "top_p": 0.9, "tokens": 200}
history = ""

print("Chat ready!  /help for commands\n")

while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break

    if not user:
        continue

    if user == "/help":
        print("""
  /reset           clear history
  /set temp  X     temperature  (default 0.8)
  /set top_p X     nucleus cutoff (default 0.9)
  /set tokens X    max new tokens (default 200)
  /quit
""")
        continue
    if user == "/reset":
        history = ""; print("  [cleared]\n"); continue
    if user == "/quit":
        print("Bye!"); break
    if user.startswith("/set "):
        try:
            _, key, val = user.split()
            cfg[key] = float(val) if key != "tokens" else int(val)
            print(f"  [{key} = {val}]\n")
        except:
            print("  Usage: /set temp|top_p|tokens VALUE\n")
        continue

    # build prompt: rolling history + new turn
    prompt   = history + f"User: {user}\nBot:"
    prompt   = prompt[-(SEQ_LEN * 4):]          # rough char trim before encoding

    reply    = generate(prompt, cfg["tokens"], cfg["temp"], cfg["top_p"]).split("User:")[0].strip()
    history += f"User: {user}\nBot: {reply}\n"

    print(f"\nBot: {reply}\n")
    