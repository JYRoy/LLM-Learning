from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import *
from model.model import *

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_SIZE = 64
HIDDEN_SIZE = 32
EPOCHS = 1
BATCH_SIZE = 2
NUM_HEADS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Prepare dataset")
    dataset = IMDBBertDataset("../data/IMDB/IMDB Dataset.csv")

    print("Prepare model")
    model = BERT(len(dataset.vocab), EMBEDDING_SIZE, HIDDEN_SIZE, NUM_HEADS).to(device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    token_criterion = nn.NLLLoss(ignore_index=0).to(device)
    nsp_cirterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.015)
    print("Begin Training")
    for i, value in tqdm(enumerate(dataloader)):
        inp, mask, inverse_token_mask, token_target, nsp_target = value
        optimizer.zero_grad()

        token, nsp = model(inp, mask)
        tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
        token = token.masked_fill(tm, 0)

        loss_token = token_criterion(token.transpose(1, 2), token_target)
        loss_nsp = nsp_cirterion(nsp, nsp_target)

        loss = loss_token + loss_nsp
        loss.backward()
        optimizer.step()

        print("Step ", i, " loss: ", loss)