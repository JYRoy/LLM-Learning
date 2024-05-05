import math
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn

from embedding import Embeddings


class PositionEncoding(nn.Module):
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


# vocab = 1000
# d_model = 512
# dropout = 0.1
# max_len = 60

# x = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 5, 6, 7]]))
# emb = Embeddings(vocab=vocab, d_model=d_model)
# embr = emb(x)
# x = embr

# pe = PositionEncoding(d_model, dropout, max_len)
# pe_result = pe(x)

# print(pe_result)
# print(pe_result.shape)
