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
src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
tgt_vocab_size = len(vocab_transform[TGT_LANGUAGE])

model = make_model(
    src_vocab_size,
    src_vocab_size,
    N=nlayers,
    d_model=emsize,
    d_ff=nhid,
    head=nhead,
    dropout=dropout,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

source_mask = Variable(torch.ones(1, 1, batch_size)).to("cuda")


def train_epoch():
    model.train()
    model.to("cuda")
    losses = 0
    train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(
        train_iter, batch_size=batch_size, collate_fn=collate_fn
    )

    for src, tgt in train_dataloader:
        src = src.to("cuda")
        tgt = tgt.to("cuda")

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(src, tgt_input, src_mask, tgt_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        print("loss: " + str(loss))

    return losses / len(list(train_dataloader))


train_epoch()
