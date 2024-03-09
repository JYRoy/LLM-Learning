import math
import torch
from torch.autograd import Variable
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        """
        vocab: the number of tokens
        d_model: the dimension of word embedding
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# vocab = 1000
# d_model = 512


# x = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 5, 6, 7]]))

# emb = Embeddings(vocab=vocab, d_model=d_model)
# embr = emb(x)
# print("embr: ", embr)
# print(embr.shape)
