import torch
import torch.nn as nn
import torch.nn.functional as F

from sublayer_connection import *
from attention import *


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Encoder layer consists of two sublayers.
        First sublayer is a Multi-head attention with residual.
        Second sublayer is a feed forward with residual.
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # first layer
        return self.sublayer[1](x, self.feed_forward)  # second layer


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer: encoder layer
        N: numbers of encoder layer
        """
        super(Encoder, self).__init__()
        self.layers = [layer for _ in range(N)]
        self.norm = LayerNorm(layer.size)  # used after Nth layers

    def forward(self, x, mask):
        """loop through N layers and norm the output"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
