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

NUM_LAYERS = 12
EMBEDDING_SIZE = 512
HIDDEN_SIZE = 768
EPOCHS = 10
BATCH_SIZE = 1
NUM_HEADS = 8
LOG_DIR = "./log"
CHECKPOINT = "./ckpt/bert_final.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Prepare dataset")
    dataset = IMDBBertDataset("../data/IMDB/IMDB Dataset.csv")
    print("vocab size:", len(dataset.vocab))

    print("Prepare model")
    model = BERT(
        NUM_LAYERS, len(dataset.vocab), EMBEDDING_SIZE, HIDDEN_SIZE, NUM_HEADS
    ).to(device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    token_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    nsp_cirterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.015)
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

    # print("Begin Training")
    # model.train()
    # for epoch in range(start_epoch, EPOCHS):
    #     start_time = time.time()
    #     print("Epoch %d 的学习率：%f" % (epoch, optimizer.param_groups[0]["lr"]))
    #     for index, value in enumerate(dataloader):
    #         (
    #             masked_sentence,
    #             inp,
    #             mask,
    #             inverse_token_mask,
    #             original_target,
    #             token_target,
    #             nsp_target,
    #         ) = value
    #         optimizer.zero_grad()

    #         token, nsp = model(inp, mask)
    #         tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
    #         token = token.masked_fill(tm, 0)

    #         loss_token = token_criterion(token.transpose(1, 2), token_target)
    #         loss_nsp = nsp_cirterion(nsp, nsp_target)

    #         loss = loss_token + loss_nsp

    #         loss.backward()
    #         optimizer.step()

    #         end_time = time.time() - start_time
    #         if index % 10 == 0:
    #             elapsed = time.gmtime(end_time)
    #             print(
    #                 f"Epoch {epoch} | {index} / {batched_length} | NSP loss {loss_nsp} | Token loss {loss_token} | elapsed {time.strftime('%H:%M:%S', elapsed)}"
    #             )
    #         if index % 2000 == 0:
    #             scheduler.step()
    #         writer.add_scalar("NSP Loss", loss_nsp, index * (epoch + 1))
    #         writer.add_scalar("TOKEN Loss", loss_token, index * (epoch + 1))
        
    #     torch.save(
    #         {
    #             "epoch": epoch,
    #             "model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "loss": loss,
    #         },
    #         CHECKPOINT,
    #     )
    #     print(f"Model save to {CHECKPOINT}!")

    print("Begin Evaluation")
    model.eval()
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
        token, nsp = model(inp, mask)
        print("Mask sentence: ", masked_sentence)
        print("Original sentence: ", original_target)
        token_softmax = torch.nn.functional.softmax(token, dim=-1)
        top_k_pred = torch.argmax(token_softmax, dim=-1)
        for k in range(1):
            pred_k = top_k_pred[:, :]
            print(pred_k)
            for each_token in pred_k[0]:
                word = dataset.vocab.lookup_token(each_token)
                print(word)
