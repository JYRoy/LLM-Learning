import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    """deep copy modules

    module: nn.Module
    N: numbers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        output = self.decode(memory, source_mask, target, target_mask)
        return self.generator(output)

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
        nn.Sequential(
            Embeddings(vocab_size=source_vocab, emb_size=d_model), c(position)
        ),
        nn.Sequential(
            Embeddings(vocab_size=target_vocab, emb_size=d_model), c(position)
        ),
        Generator(d_model, target_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward"""

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
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        """
        layer: encoder layer
        N: numbers of encoder layer
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  # used after Nth layers

    def forward(self, x, mask):
        """loop through N layers and norm the output"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module"""

    def __init__(self, features, eps=1e-6) -> None:
        """ "
        features: word embedding dimension
        eps: a minimum value is used to prevent the denominator from being 0
        """
        super(LayerNorm, self).__init__()

        # two trainable parameters
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

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


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward"""

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
    """Generic N layer decoder with masking."""

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


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'

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


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: word embedding dimension,
            it is the input dimension of first linear
            and output dimension of the second linear
        d_ff: middle dimensional between two linears
        dropout: dropout ratio
        """
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x is the output of last layer, dimension is (d_model)
        """

        return self.w2(self.dropout(F.relu(self.w1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.emb_size)


class PositionEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        """
        d_model: the dimension of word embedding
        dropout: probility
        max_len: max length of sequence
        """
        super(PositionEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # position encoding matrix, size is (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # absolute position encoding
        position = torch.arange(0, max_len).unsqueeze(1)  # size is (max_len, 1)

        # sin or cos functions of differnent frequencies which is same as paper description
        i = torch.arange(0, d_model, 2)
        div = 10000.0 ** (2 * i / d_model)
        term = position / div
        pe[:, 0::2] = torch.sin(term)
        pe[:, 1::2] = torch.cos(term)

        # show position encoding
        # plt.figure(1)
        # plt.imshow(pe.detach().numpy(), cmap="jet", interpolation="nearest")
        # plt.savefig('.images/position_encoding.jpg',bbox_inches='tight')
        # plt.show()

        pe = pe.unsqueeze(-2)

        # position is not the trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: token embedding from embedding module
        """
        # resize max_len dimension to x length
        x = x + Variable(self.pe[: x.size(0), :], requires_grad=False)
        return self.dropout(x)
