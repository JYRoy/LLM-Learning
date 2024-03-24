import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(size):
    """mask subsequent tensor

    size: the size of last two dimensions of the tensor
    """
    attn_shape = (1, size, size)
    # upper triangular matrix
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    # lower triangular matrix
    # 1 means masked, 0 means unmasked
    # row means the current position
    # col means the related postions with current position
    # for example: the index 2(3 position) could see 2 tokens
    return torch.from_numpy(1 - subsequent_mask)


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
    p_atten = F.softmax(scores, dim=-1)

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
            model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)  # [batch_size, head(8), seq_len, 64]
            for model, x in zip(self.linears, (query, key, value))
        ]

        # calculate attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # re-transpose 1, 2 dimension
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)  # [batchsize, seq_len, 512]

        # last linear transpose
        return self.linears[-1](x)  # [batchsize, seq_len, 512]