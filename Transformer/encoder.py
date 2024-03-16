import numpy as np
import matplotlib.pyplot as plt
import math
import torch
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
