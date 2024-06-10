import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import *
from model.model import *

BASE_DIR = Path(__file__).resolve().parent

NUM_LAYERS = 1
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 36
EPOCHS = 4
BATCH_SIZE = 12
NUM_HEADS = 4
LOG_DIR = "./log"
CHECKPOINT = "./ckpt/bert_final.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def percentage(batch_size: int, max_index: int, current_index: int):
    """Calculate epoch progress percentage

    Args:
        batch_size: batch size
        max_index: max index in epoch
        current_index: current index

    Returns:
        Passed percentage of dataset
    """
    batched_max = max_index
    return round(current_index / batched_max * 100, 2)


def nsp_accuracy(result: torch.Tensor, target: torch.Tensor):
    """Calculate NSP accuracy between two tensors

    Args:
        result: result calculated by model
        target: real target

    Returns:
        NSP accuracy
    """
    s = (
        result.argmax(1) == target.argmax(1)
    ).sum()  # ([1, 1, 0, 0, ..., 1] == [1, 0, 0, 0, ..., 1]).sum()
    return round(float(s / result.size(0)), 2)


def token_accuracy(
    result: torch.Tensor, target: torch.Tensor, inverse_token_mask: torch.Tensor
):
    """Calculate MLM accuracy between ONLY masked words

    Args:
        result: result calculated by model
        target: real target
        inverse_token_mask: well-known inverse token mask

    Returns:
        MLM accuracy
    """
    r = result.argmax(-1).masked_select(~inverse_token_mask)
    t = target.masked_select(~inverse_token_mask)
    s = (r == t).sum()
    return round(float(s / (result.size(0) * result.size(1))), 2)


if __name__ == "__main__":
    print("Prepare dataset")
    dataset = IMDBBertDataset("../data/IMDB/IMDB Dataset.csv")
    print("vocab size:", len(dataset.vocab))

    print("Prepare model")
    model = BERT(
        NUM_LAYERS, len(dataset.vocab), EMBEDDING_SIZE, HIDDEN_SIZE, NUM_HEADS
    ).to(device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    token_criterion = nn.NLLLoss(ignore_index=0).to(device)
    nsp_cirterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    batched_length = len(dataset) // BATCH_SIZE

    writer = SummaryWriter(LOG_DIR)

    if os.path.exists(CHECKPOINT):
        print("Loading checkpoint")
        checkpoint = torch.load(CHECKPOINT)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("No checkpoint")
        start_epoch = 0

    print("Begin Training")
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        print("Epoch %d 的学习率：%f" % (epoch, optimizer.param_groups[0]["lr"]))
        for index, value in enumerate(dataloader):
            (
                masked_sentence,
                inp,
                mask,
                inverse_token_mask,
                original_target,
                token_target,
                nsp_target,
            ) = value
            optimizer.zero_grad()

            token, nsp = model(inp, mask)
            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            token = token.masked_fill(tm, 0)

            loss_token = token_criterion(token.transpose(1, 2), token_target)
            loss_nsp = nsp_cirterion(nsp, nsp_target)

            loss = loss_token + loss_nsp

            loss.backward()
            optimizer.step()

            end_time = time.time() - start_time
            if index % 10 == 0:
                elapsed = time.gmtime(end_time)
                global_step = epoch * len(dataloader) + index
                nsp_acc = nsp_accuracy(nsp, nsp_target)
                token_acc = token_accuracy(token, token_target, inverse_token_mask)

                writer.add_scalar(
                    "NSP train accuracy", nsp_acc, global_step=global_step
                )
                writer.add_scalar(
                    "Token train accuracy", token_acc, global_step=global_step
                )
                acc_res = f" | NSP accuracy {nsp_acc} | Token accuracy {token_acc}"

                print(
                    f"Epoch {epoch} | {index} / {len(dataloader)} {percentage(BATCH_SIZE, len(dataloader), index)}% | NSP loss {loss_nsp} | Token loss {loss_token} | elapsed {time.strftime('%H:%M:%S', elapsed)} | "
                    + acc_res
                )
            writer.add_scalar("NSP Loss", loss_nsp, index * (epoch + 1))
            writer.add_scalar("TOKEN Loss", loss_token, index * (epoch + 1))
        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            CHECKPOINT,
        )
        print(f"Model save to {CHECKPOINT}!")
