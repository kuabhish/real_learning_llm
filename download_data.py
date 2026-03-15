"""
Run this once before training.
Downloads real conversation data and saves to data.txt
"""
from datasets import load_dataset
import os

DATA_PATH   = "data.txt"
TARGET_SIZE = 100_000_000   # 30MB is plenty to start

if os.path.exists(DATA_PATH):
    print("data.txt already exists — delete it first if you want to redownload")
    exit()

text   = ""
chunks = []

print("downloading UltraChat...")

# ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:20000]")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:20000]")

for row in ds:
    convo = row["messages"]
    lines = []

    for msg in convo:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'].strip()}")

    chunks.append("\n".join(lines) + "\n\n")

print(f"  got {len(ds)} conversations")

# # ── 1. Alpaca — 52k instruction/answer pairs ──────────────────────────────────
# print("downloading Alpaca...")
# ds = load_dataset("tatsu-lab/alpaca", split="train")
# for row in ds:
#     q = row["instruction"].strip()
#     if row["input"].strip():
#         q += "\n" + row["input"].strip()
#     a = row["output"].strip()
#     chunks.append(f"Human: {q}\nAssistant: {a}\n\n")
# print(f"  got {len(ds)} examples")

# # ── 2. DailyDialog — 13k natural conversations ────────────────────────────────
# print("downloading DailyDialog...")
# ds = load_dataset("conv_ai_2", split="train")
# for convo in ds:
#     dialog = convo["dialog"]
#     lines = []
#     for i, line in enumerate(dialog):
#         role = "Human" if i % 2 == 0 else "Assistant"
#         lines.append(f"{role}: {line.strip()}")
#     chunks.append("\n".join(lines) + "\n\n")

# print(f"  got {len(ds)} conversations")

# # ── 3. OpenAssistant — real human↔AI conversations ───────────────────────────
# print("downloading OpenAssistant...")
# ds       = load_dataset("OpenAssistant/oasst1", split="train")
# by_id    = {r["message_id"]: r for r in ds}
# roots    = [r for r in ds if r["parent_id"] is None and r["lang"] == "en"]
# for root in roots:
#     thread, node = [], root
#     while node:
#         role = "Human" if node["role"] == "prompter" else "Assistant"
#         thread.append(f"{role}: {node['text'].strip()}")
#         children = [r for r in ds if r["parent_id"] == node["message_id"]]
#         node = children[0] if children else None
#     if len(thread) >= 2:
#         chunks.append("\n".join(thread) + "\n\n")
# print(f"  got {len(roots)} threads")

# ── combine and repeat to hit target size ─────────────────────────────────────
import random
random.shuffle(chunks)
text = "".join(chunks)

while len(text) < TARGET_SIZE:
    random.shuffle(chunks)
    text += "".join(chunks)

text = text[:TARGET_SIZE]

with open(DATA_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nsaved {len(text)/1e6:.1f}MB to data.txt")
print("\npreview:")
print(text[:400])
print("\nnow run: python mind_palace_llm.py")
