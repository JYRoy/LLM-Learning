import time
import tiktoken
from model import *
from dataloader import *


def toy_eval():
    num_return_sequences = 5
    max_length = 30
    torch.set_float32_matmul_precision("medium")
    # model = GPT.from_pretrained("gpt2")
    model = GPT(GPTConfig())
    model.train()
    model.to("cuda")
    model = torch.compile(model)

    # get a data batch
    train_loader = DataLoaderLite(B=4, T=1024)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(
            f"step: {i}, loss: {loss.item()}, dt: {dt:.2f}ms,, tok/sec: {tokens_per_sec:.2f}"
        )

    exit(-1)

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
            ix = torch.multinomial(
                topk_probs, 1
            )  # (B, 1), ix is the index of topk_probs
            xcol = torch.gather(
                topk_indices, -1, ix
            )  # (B, 1), get the token indice by the topk_probs' index
            x = torch.cat(
                (x, xcol), dim=1
            )  # concat the original x with the predicted token indice xcol

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    toy_eval()
