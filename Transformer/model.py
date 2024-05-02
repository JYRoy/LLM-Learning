import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *
from decoder import *
from embedding import *
from encoder import *
from position_encoding import *
from feed_forward import *
from norm import *
from output import *
from sublayer_connection import *

source_vocab = 11
target_vocab = 11
N = 6

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encode(source, source_mask)
        return self.decode(memory, source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(
    source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1
):
    """make transformer model

    Args:
        source_vocab (int): source language vocab size
        target_vocab (int): target language vocab size
        N (int, optional): number of encoders and decoders. Defaults to 6.
        d_model (int, optional): word embedding dimension size. Defaults to 512.
        d_ff (int, optional): feed forward dimension. Defaults to 2048.
        head (int, optional): head numbers. Defaults to 8.
        dropout (float, optional): dropout ratio. Defaults to 0.1.
    """
    c = copy.deepcopy

    attn = MultiHeadedAttention(head, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

if __name__=='__main__':
    model = make_model(source_vocab, target_vocab, N)
    print(model)