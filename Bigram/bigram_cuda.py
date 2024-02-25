import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32  # B
block_size = 8  # T means time
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(2024)

with open("../data/tiny_shakespeare/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create two maps from characters in the dataset
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # string to integers
decode = lambda l: "".join([itos[i] for i in l])  # integers to string

# split train and val data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )  # num_embedding, embedding_dim

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(
            idx
        )  # logits: (B, T, C)  idx & targets: (B, T)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Reshape logits and targets for loss calculation, see notes for explanation
            logits = logits.view(B * T, C)  # (32, 8, 65) -> (256, 65)
            targets = targets.view(B * T)  # (32, 8) -> (256)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # id shape: (B, T)
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C), also take the last one as predict token
            probs = F.softmax(logits, dim=-1)  # (B, C), Convert logits to probabilities
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B, 1), Sample the next token
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # (B, T+1), Append the next token to the sequence
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
