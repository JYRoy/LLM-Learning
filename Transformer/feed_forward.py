import torch
import torch.nn as nn
import torch.nn.functional as F


class PosotionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: word embedding dimension,
            it is the input dimension of first linear
            and output dimension of the second linear
        d_ff: middle dimensional between two linears
        dropout: dropout ratio
        """
        super(PosotionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x is the output of last layer, dimension is (d_model)
        """

        return self.w2(self.dropout(F.relu(self.w1(x))))
