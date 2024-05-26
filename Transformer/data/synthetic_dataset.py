import torch
from torch.autograd import Variable
import numpy as np

from .common import *


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))  # (batch_size, seq_len)
        data[:, 0] = 1  # change the first token to 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
