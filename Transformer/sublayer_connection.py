import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import *


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        size: word embedding dimension
        dropout: dropout ratio
        """
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))
