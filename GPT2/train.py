import os
import time
import tiktoken
from model import *
from dataloader import *

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# initialize DDP
ddp = int(os.environ.get("RANK", -1)) != -1  # global rank
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = f"cuda:{ddp_local_rank}"
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")


max_lr = 3e-4
min_lr = max_lr * 0.1
max_steps = 50
warmup_steps = 10
num_return_sequences = 5
max_length = 30
total_batch_size = 524288  # 2**19，~0.5M batch size in tokens
B = 4  # micro batch size
T = 1024  # sequence length
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"  # it is distibuted data parallel so that batch on each device is a part of the micro batch
grad_accum_steps = total_batch_size // (
    B * T * ddp_world_size
)  # we need less grad accumulate step on ddp mode comparing with single gpu mode
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
print("I am GPU ", ddp_rank)
torch.set_float32_matmul_precision("medium")


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig(vocab_size=50304))
model.train()
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# get a data batch
train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size
)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizer(
    weight_decay=0.1, learning_rate=6e-4, device=device
)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == grad_accum_steps - 1
            )  # only on the last step, model needs sync its grads on ddp mode. This behaviour is same as with no_sync context manager.
        loss.backward()
    if ddp:
        dist.all_reduce(
            loss_accum, op=dist.ReduceOp.AVG
        )  # just collect average loss for all processes and print it on the main process
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )

if ddp:
    destroy_process_group()

exit(0)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        logits = logits[
            :, -1, :
        ]  # take the logits at the last position (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (5, 50)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1), ix is the index of topk_probs
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
