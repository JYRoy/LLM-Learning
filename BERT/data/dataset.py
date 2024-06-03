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

"""example
[25490 rows x 4 columns]
                                          masked_indices                                            indices                                         token_mask  is_next
0      [0, 5, 6, 7, 8, 9, 6087, 11, 12, 1296, 14, 15,...  [0, 5, 6, 7, 8, 9, 6087, 11, 12, 1296, 14, 15,...  [True, True, True, True, True, True, False, Tr...        1
1      [0, 28, 2, 2268, 27, 111, 20, 302, 465, 2, 335...  [0, 28, 2, 2268, 27, 111, 20, 302, 465, 2, 335...  [True, True, False, True, True, True, True, Tr...        0
2      [0, 24, 25, 26, 27, 28, 2, 2, 31, 32, 33, 34, ...  [0, 24, 25, 26, 27, 28, 2, 2, 31, 32, 33, 34, ...  [True, True, True, True, True, True, False, Fa...        1
3      [0, 2, 2, 121, 6375, 7, 9004, 57, 2642, 207, 7...  [0, 2, 2, 121, 6375, 7, 9004, 57, 2642, 207, 7...  [True, False, False, True, True, True, False, ...        0
4      [0, 7, 36, 37, 2, 38, 2, 39, 2, 40, 41, 42, 43...  [0, 7, 36, 37, 2, 38, 2, 39, 2, 40, 41, 42, 43...  [True, True, True, True, False, True, False, T...        1
...                                                  ...                                                ...                                                ...      ...
25485  [0, 34, 2164, 2, 6, 4087, 4, 1419, 6296, 81, 3...  [0, 34, 2164, 2, 6, 4087, 4, 1419, 6296, 81, 3...  [True, True, True, False, True, False, True, T...        0
25486  [0, 571, 3510, 6748, 25, 83, 4581, 2, 9520, 57...  [0, 571, 3510, 6748, 25, 83, 4581, 2, 9520, 57...  [True, True, True, True, True, True, True, Fal...        1
25487  [0, 24, 3560, 27, 24, 802, 27, 90, 55, 477, 85...  [0, 24, 3560, 27, 24, 802, 27, 90, 55, 477, 85...  [True, True, True, True, True, True, True, Tru...        0
25488  [0, 571, 462, 4224, 2, 7059, 2782, 25, 6607, 1...  [0, 571, 462, 4224, 2, 7059, 2782, 25, 6607, 1...  [True, True, True, True, False, False, True, T...        1
25489  [0, 55, 4, 2, 2, 55, 1253, 30, 2778, 57, 55, 2...  [0, 55, 4, 2, 2, 55, 1253, 30, 2778, 57, 55, 2...  [True, True, True, False, False, True, True, T...        0
"""
class IMDBBertDataset(Dataset):

    def __init__(self, path):
        self.columns = [
            MASKED_INDICES_COLUMN,
            TARGET_COLUMN,
            TOKEN_MASK_COLUMN,
            NSP_TARGET_COLUMN,
        ]
        self.ds = pd.read_csv(path)["review"]
        self.ds = self.ds[0:1000]  # cut the dataset
        self.tokenizer = get_tokenizer("basic_english")
        self.optimal_sentence_length = None
        self.counter = Counter()  # token counter map, {"AAA": 2, "BB": 1}
        self.vocab = None
        self.df = self.prepare_dataset()

    def prepare_dataset(self):
        sentences = []
        nsp = []  # next sentence predict
        sentence_lens = []
        # split dataset on sentences
        print("Get all sentences")
        for review in tqdm(self.ds):
            review_sentences = review.split(".")  # split a review into serval sentences
            sentences += review_sentences  # collect all sentensens of reviewes
            for v in sentences:
                sentence_len = len(v.split())
                sentence_lens.append(sentence_len)  # collect length of each sequence
        sentence_lens = np.array(sentence_lens)
        self.optimal_sentence_length = int(
            np.percentile(sentence_lens, OPTIMAL_LENGTH_PERCENTILE)
        )  # get the value of length percentile, e.g. 27 in IMDB

        print("Create vocabulary")
        for sentence in tqdm(sentences):
            tok = self.tokenizer(sentence)  # list
            self.counter.update(tok)

        self._fill_vocab()

        print("Preprocessing dataset")
        for review in tqdm(self.ds):
            review_sentences = review.split(".")
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    # positive sample: next sentences pair
                    first, second = self.tokenizer(review_sentences[i]), self.tokenizer(
                        review_sentences[i + 1]
                    )
                    nsp.append(self._create_item(first, second, 1))
                    # negative sample: random sentences pair
                    first, second = self._select_false_nsp_sentences(sentences)
                    first, second = self.tokenizer(first), self.tokenizer(second)
                    nsp.append(self._create_item(first, second, 0))
        return pd.DataFrame(nsp, columns=self.columns)

    def _create_item(self, first, second, target):
        # mask and pad original sentences
        padded_masked_first_sentence, first_mask = self._preprocess_sentence(
            first, True
        )
        padded_masked_second_sentence, second_mask = self._preprocess_sentence(
            second, True
        )

        # create nsp sentence pair
        padded_masked_nsp_sentence = (
            padded_masked_first_sentence + [SEP] + padded_masked_second_sentence
        )
        padded_masked_nsp_sentence_indices = self.vocab.lookup_indices(
            padded_masked_nsp_sentence
        )
        inverse_token_mask = first_mask + [True] + second_mask

        original_fisrt, _ = self._preprocess_sentence(first, False)
        original_second, _ = self._preprocess_sentence(second, False)
        original_nsp_sentence = original_fisrt + [SEP] + original_second
        original_nsp_sentence_indices = self.vocab.lookup_indices(original_nsp_sentence)

        """
        self.columns = [
            MASKED_INDICES_COLUMN,
            TARGET_COLUMN,
            TOKEN_MASK_COLUMN,
            NSP_TARGET_COLUMN,
        ]
        """
        return (
            padded_masked_nsp_sentence_indices,
            original_nsp_sentence_indices,
            inverse_token_mask,
            target,
        )

    def _select_false_nsp_sentences(self, sentences):
        sentences_len = len(sentences)
        sentences_index = random.randint(0, sentences_len - 1)
        next_sentences_index = random.randint(0, sentences_len - 1)

        while next_sentences_index == sentences_index + 1:
            next_sentences_index = random.randint(0, sentences_len - 1)

        return sentences[sentences_index], sentences[next_sentences_index]

    def _preprocess_sentence(self, sentence, should_mask):
        inverse_token_mask = None
        if should_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
            padded_masked_sentence, inverse_token_mask = self._pad_sentence(
                [CLS] + sentence, [True] + inverse_token_mask
            )
            return padded_masked_sentence, inverse_token_mask
        else:
            return self._pad_sentence([CLS] + sentence, [True])

    def _mask_sentence(self, sentence):
        """
        replace 15% word with [MASK]
        """
        sentence_length = len(sentence)
        inverse_token_mask = [
            True for _ in range(max(sentence_length, self.optimal_sentence_length))
        ]  # When token is unmasked, set True

        mask_amount = round(
            sentence_length * MASK_PERCENTAGE
        )  # only 15% tokens are masked
        for _ in range(mask_amount):
            idx = random.randint(0, sentence_length - 1)
            if random.random() < 0.8:  # 80% set [MASK] token
                sentence[idx] = MASK
            else:  # otherwise set random word from the vocabulary
                other_token_index = random.randint(
                    5, len(self.vocab) - 1
                )  # start from 5 cause expecting special token
                sentence[idx] = self.vocab.lookup_token(other_token_index)
            inverse_token_mask[idx] = False  # When token is masked, set False
        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence, inverse_token_mask):
        sentence_length = len(sentence)
        if sentence_length >= self.optimal_sentence_length:
            new_sentence = sentence[:sentence_length]
        else:
            new_sentence = sentence + [PAD] * (
                self.optimal_sentence_length - sentence_length
            )

        if inverse_token_mask:
            mask_length = len(inverse_token_mask)
            if mask_length >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[: self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (
                    self.optimal_sentence_length - mask_length
                )
        return new_sentence, inverse_token_mask

    def _fill_vocab(self):
        self.vocab = vocab(self.counter, min_freq=2)
        self.vocab.insert_token(CLS, 0)
        self.vocab.insert_token(PAD, 1)
        self.vocab.insert_token(MASK, 2)
        self.vocab.insert_token(SEP, 3)
        self.vocab.insert_token(UNK, 4)
        self.vocab.set_default_index(4)

    def __len__(self):
        return len(self.df)

    def __get_item__(self, idx):
        item = self.df.iloc[idx]
        masked_indices = torch.Tensor(item[MASKED_INDICES_COLUMN]).long()
        token_mask = torch.Tensor(item[TOKEN_MASK_COLUMN]).bool()
        target_indices = torch.Tensor(item[TARGET_COLUMN]).long()
        target_indices = target_indices.masked_fill_(
            token_mask, 0
        )  # only save the masked token indices, unmasked intergers in target to 0, we only want the model to predict masked tokens

        attention_mask = (masked_indices == self.vocab[PAD]).unsqueeze(0)

        # NSP task is binary classificatio problem, will use BCEWithLogitsLoss
        if item[NSP_TARGET_COLUMN] == 0:  # todo
            t = [1, 0]  # not next sentence
        else:
            t = [0, 1]  # next sentence

        nsp_target = torch.Tenosr(t)
        return (
            masked_indices.cuda(),
            attention_mask.cuda(),
            token_mask.cuda(),
            target_indices.cuda(),
            nsp_target.cuda(),
        )


dataset = IMDBBertDataset("../../data/IMDB/IMDB Dataset.csv")
