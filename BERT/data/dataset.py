import random
import typing
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

CLS = "[CLS]"
PAD = "[PAD]"
SEP = "[SEP]"
MASK = "[MASK]"
UNK = "[UNK]"
MASK_PERCENTAGE = 0.15  # How much words to mask
MASKED_INDICES_COLUMN = "masked_indices"
TARGET_COLUMN = "indices"
NSP_TARGET_COLUMN = "is_next"
TOKEN_MASK_COLUMN = "token_mask"
OPTIMAL_LENGTH_PERCENTILE = 70


class IMDBBertDataset(Dataset):

    def __init__(self, path):
        self.ds = pd.read_csv(path)["review"]
        self.tokenizer = get_tokenizer("basic_english")
        self.optimal_sentence_length = None
        self.counter = Counter()  # token counter map, {"AAA": 2, "BB": 1}
        self.vocab = None
        self.df = self.prepare_dataset()

    def prepare_dataset(self):

        sentences = []
        nsp = []
        sentence_lens = []
        # split dataset on sentences
        print("Get all sentences ")
        # i = 0
        for review in tqdm(self.ds):
            review_sentences = review.split(".")  # split a review into serval sentences
            sentences += review_sentences  # collect all sentensens of reviewes
            for v in sentences:
                sentence_len = len(v.split())
                sentence_lens.append(sentence_len)  # collect length of each sequence
            # if i > 10:
            #     break
            # i += 1
        sentence_lens = np.array(sentence_lens)
        self.optimal_sentence_length = int(
            np.percentile(sentence_lens, OPTIMAL_LENGTH_PERCENTILE)
        )  # get the value of length percentile, e.g. 27 in IMDB

        print("Create vocabulary")
        for sentence in tqdm(sentences):
            tok = self.tokenizer(sentence)  # list
            self.counter.update(tok)

    def __len__(self):
        return len(self.df)

    def _fill_vocab(self):
        self.vocab = vocab(self.counter, min_freq=2)
        self.vocab.insert_token(CLS, 0)
        self.vocab.insert_token(PAD, 1)
        self.vocab.insert_token(MASK, 2)
        self.vocab.insert_token(SEP, 3)
        self.vocab.insert_token(UNK, 4)
        self.vocab.set_default_index(4)


dataset = IMDBBertDataset("../../data/IMDB/IMDB Dataset.csv")
