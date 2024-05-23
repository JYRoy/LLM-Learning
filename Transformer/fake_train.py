from pyitcast.transformer_utils import (
    get_std_opt,
    LabelSmoothing,
    SimpleLossCompute,
    run_epoch,
    greedy_decode,
    TransformerModel
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from data.multi30k_dataset import *

V = 512
batch_size = 20
num_batch = 30

model = make_model(V, V, N=2)

model_optimizer = get_std_opt(model)

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


def run(model, loss, epochs=10):
    model.to("cuda")
    for _ in range(epochs):
        model.train()
        run_epoch(fake_data_generator(V, batch_size, num_batch), model, loss)
        model.eval()
        run_epoch(fake_data_generator(V, batch_size, num_batch), model, loss)

    model.eval()

    source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]])).to("cuda")

    source_mask = Variable(torch.ones(1, 1, 10)).to("cuda")

    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == "__main__":
    run(model, loss, 20)
