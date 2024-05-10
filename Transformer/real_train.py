from pyitcast.transformer_utils import (
    get_std_opt,
    LabelSmoothing,
    SimpleLossCompute,
    run_epoch,
    greedy_decode,
)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from data_generator import *
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emsize = 512
batch_size = 128
num_batch = 30
nhid = 2048
nhead = 8
nlayers = 6
dropout = 0.2
lr = 0.5
bptt = 1024
log_interval = 200

model = make_model(
    len(vocab_src),
    len(vocab_tgt),
    N=nlayers,
    d_model=emsize,
    d_ff=nhid,
    head=nhead,
    dropout=dropout,
)

criterion = nn.CrossEntropyLoss(ignore_index=-1).to("cuda")
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

source_mask = Variable(torch.ones(1, 1, batch_size)).to("cuda")

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

def train_epoch():
    pad_idx = vocab_tgt["<blank>"]
    model.train()
    model.to("cuda")
    losses = 0
    train_dataloader, valid_dataloader = create_dataloaders(
        "cuda",
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=batch_size,
        max_padding=MAX_LEN,
        is_distributed=False,
    )

    train_iter = (Batch(b[0], b[1], pad_idx) for b in train_dataloader)
    for i, batch in enumerate(train_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        optimizer.zero_grad()

        loss = criterion(out.reshape(-1, out.shape[-1]), batch.tgt_y.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        print("loss: ", loss)

    return losses / len(list(train_dataloader))


train_epoch()
