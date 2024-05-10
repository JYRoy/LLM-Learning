import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_generator import *


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((1, sz, sz), device="cuda")) == 1).transpose(1, 2)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):  # (batch_size, seq_len)
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((1, src_seq_len, src_seq_len), device="cuda").type(torch.bool)
    src_padding_mask = (src == PAD_IDX).unsqueeze(-2)
    tgt_padding_mask = (tgt == PAD_IDX).unsqueeze(-2)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def attention(query, key, value, mask=None, dropout=None):
    """
    query, key, value: attention tensor
    mask: mask tensor
    dropout: Dropout instance
    """
    d_k = query.size(-1)  # get token embedding size

    # calculate scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # mask fill
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)

    # softmax on scores
    p_atten = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_atten = dropout(p_atten)

    # calculate attention value
    attn_val = torch.matmul(p_atten, value)
    return attn_val, p_atten


def clones(module, N):
    """deep copy modules

    module: nn.Module
    N: numbers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        head: head numbers, 8 in paper
        embedding_dim: token embedding dimension, 512 in paper
        dropout: dropout ratio
        """
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % head == 0

        # get embedding dimension for each head, 512 // 8 = 64
        self.d_k = embedding_dim // head

        self.head = head
        self.embedding_dim = embedding_dim

        # multi-headed attention uses four linears
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # attention tensor after softmax(QK^T/d_k)
        self.attn = 0

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # expand the first dimension

        batch_size = query.size(0)  # get batch size

        # linear transform for V, K, Q
        # linear dimension equals to embedding_dim(512), but it splitted into 8 heads
        # it means that project them to a smaller space(self.d_k = 64)
        query, key, value = [
            model(x)
            .view(batch_size, -1, self.head, self.d_k)
            .transpose(1, 2)  # [batch_size, head(8), seq_len, 64]
            for model, x in zip(self.linears, (query, key, value))
        ]

        # calculate attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # re-transpose 1, 2 dimension
        x = (
            x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        )  # [seq_len, batchsize, 512]

        # last linear transpose
        return self.linears[-1](x)  # [batchsize, seq_len, 512]
