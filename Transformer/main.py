import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.functional as F
import torch.nn as nn
from torch.autograd import Variable

from embedding import *
from encoder import *
from position_encoding import *


vocab = 1000
d_model = 512
dropout = 0.1
max_len = 60

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