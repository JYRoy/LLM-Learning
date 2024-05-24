import torch
from torch.autograd import Variable
import numpy as np

from .common import *

def fake_data_generator(V, batch_size, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))
        data[:, 0] = 1

        source = Variable(data, requires_grad=False).to("cuda")
        target = Variable(data, requires_grad=False).to("cuda")

        yield Batch(source, target)