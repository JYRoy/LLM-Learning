import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
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
