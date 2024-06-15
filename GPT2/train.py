import tiktoken
from model import *

num_return_sequences = 5
max_length = 30
model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig())
model.train()
model.to("cuda")

# prefixt tokens
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(
    tokens, dtype=torch.long
)  # shape (8) -> tensor([15496,    11,   314,  1101,   257,  3303,  2746,    11])
tokens = tokens.unsqueeze(0).repeat(
    num_return_sequences, 1
)  # shape (num_return_sequences, 8) -> (B = 5, T = 8)
x = tokens.to("cuda")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        logits = logits[
            :, -1, :
        ]  # take the logits at the last position (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (5, 50)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
