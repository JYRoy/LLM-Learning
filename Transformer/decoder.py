import torch
import torch.nn as nn
import torch.nn.functional as F

from sublayer_connection import *
from attention import *


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        self_attn: multi-head self attention
        src_attn: multi-head attention
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        x: input
        memory: from encoder
        source_mask: source data mask, mask invalid word from encoder to
            increase performance and speed
        target_mask: target data mask, mask training data in encoder since
            model can not use the future words to predict next word
        """
        m = memory
        # first sublayer: Masked multi-head self attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # second sublayer: Multi-head self attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # third sublayer: feed forward
        x = self.sublayer[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer: Decoder layer
        N: numbers of Decoder layer
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  # used after Nth layers

    def forward(self, x, memory, source_mask, target_mask):
        """loop through N layers and norm the output"""
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)
