import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.functional as F
import torch.nn as nn
from torch.autograd import Variable

from attention import *
from decoder import *
from embedding import *
from encoder import *
from position_encoding import *
from feed_forward import *
from norm import *
from output import *
from sublayer_connection import *

vocab = 1000
d_model = 512
dropout = 0.1
max_len = 60
head = 8

x = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 5, 6, 7]]))
emb = Embeddings(vocab=vocab, d_model=d_model)
embr = emb(x)
x = embr

pe = PositionEncoding(d_model, dropout, max_len)
pe_result = pe(x)


size = 5
sm = subsequent_mask(size)
# print("sm: ", sm)
# plt.figure(figsize = (5, 5))
# plt.imshow(subsequent_mask(20)[0])
# plt.savefig('.images/subsequent_mask.jpg',bbox_inches='tight')
"""
output: 
sm:  tensor([[[1, 0, 0, 0, 0],
              [1, 1, 0, 0, 0]
              [1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1]]], dtype=torch.uint8)
"""

query = key = value = pe_result
mask = Variable(torch.zeros(2, 4, 4))
attn, p_attn = attention(query, key, value, mask=mask)
print(attn)
print(attn.shape)
print(p_attn)
print(p_attn.shape)

mha = MultiHeadedAttention(head, d_model, dropout)
mha_res = mha(query, key, value, mask)
print(mha_res)
print(mha_res.shape)

x = mha_res
d_ff = 64
ff = PosotionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
print(ff_result)
print(ff_result.shape)

x = ff_result
print(x)

eps = 1e-6
feature = d_model
ln = LayerNorm(feature, eps)
ln_result = ln(x)
print(ln_result)

size = d_model
x = pe_result
mask = Variable(torch.zeros(2, 4, 4))
self_attn = MultiHeadedAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)

sc = SubLayerConnection(size, dropout)
sc_result = sc(x, sublayer)
print(sc_result)
print(sc_result.shape)

c = copy.deepcopy
layer = EncoderLayer(size, self_attn, ff, dropout)
N = 8

en = Encoder(layer, N)
en_res = en(x, mask)
print(en_res)
print(en_res.shape)


self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
ff = PosotionwiseFeedForward(d_model, d_ff, dropout)

x = pe_result
memory = en_res
mask = Variable(torch.zeros(2, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)

de = Decoder(dl, N)
de_res = de(x, memory, source_mask, target_mask)
print(de_res)
print(de_res.shape)

gen = Generator(d_model, vocab)
gen_result = gen(x)
print(gen_result)
print(gen_result.shape)