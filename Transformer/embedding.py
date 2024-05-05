import math
import torch
from torch.autograd import Variable
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# vocab = 1000
# d_model = 512


# x = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 5, 6, 7]]))

# emb = Embeddings(vocab=vocab, d_model=d_model)
# embr = emb(x)
# print("embr: ", embr)
# print(embr.shape)
