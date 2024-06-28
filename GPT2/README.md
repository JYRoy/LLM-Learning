# GPT2

- [GPT2](#gpt2)
  - [Overview](#overview)
  - [Architecture](#architecture)
    - [Attention Module](#attention-module)
    - [Mathematical trick in self-attention](#mathematical-trick-in-self-attention)
  - [Zero-shot](#zero-shot)
  - [Dataset](#dataset)
  - [FlashAttention](#flashattention)
  - [Reference](#reference)


## Overview

Language Models are Unsupervised Multitask Learners

## Architecture

GPT2 still use Transformer decoder without cross multi-head attention. There are four model size in the GPT2 mini-series. The biggest one is a 1.5B parameter Transformer. We usually called the biggest one as GPT2.

| Parameters | Layers | d_model |
| ---------- | ------ | ------- |
| 124M       | 12     | 768     |
| 345M       | 24     | 1024    |
| 762M       | 36     | 1280    |
| 1542M      | 48     | 1600    |

The main two changes are:

1. Layer normalization was moved to the input of each sub-block, similar to a pre-activation residual network.
2. An additional layer normalization was added after the final self-attention block.

### Attention Module

GPT2 has a same masked multi self-attention module with GPT1. The implementation is show below.

The fomula is

- n: sequence length
- $d_k$: embedding dimension
- $O_{n\times{d_v}}$: 
$$
Attention(Q_{n\times{d_k}}, K_{n\times{d_k}}, V_{n\times{d_v}}) = mask(softmax(\frac{Q_{n\times{d_k}}K_{d_k\times{n}}^T}{\sqrt{d_k}})) V_{n\times{d_v}} = mask(Attention_{n\times{n}})V_{n\times{d_v}} = O_{n\times{d_v}}
$$

### Mathematical trick in self-attention

Lets' focusing on the mathematical trick in self-attention. It is the key point of all Transformer based model. The code of this part is on [test_self_attention.py](./test_self_attention.py)

Create a batch. We would like to couple each other on the second dimension(sequence length) in a specific way, like that the fifth location, it should not communicate with tokens in the sixth, seventh, and eight location, cause those are future tokens in the sequence. So that, information only flows from previous context to the current time step.

```python
torch.manual_seed(2024)
B, T, C = 4, 8, 2  # Batch size, sequence length (or time), embedding size (or channels)
x = torch.randn(B, T, C)  # torch.Size([4, 8, 2])
```

Before looking up self-attention, let's considering a easier way to make a communication for each other. We want $x[b, t] = mean_{i<=t} x[b, i]$ that using the mean value before current token. The `xbow[b, t]` is the vertical average of all tokens on the column.

```python
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)
"""
xbow[0]
tensor([[-1.2262, -0.0093],
        [ 0.1579, -0.2375],
        [ 0.1984, -0.2453],
        [ 0.3045, -0.4730],
        [ 0.2671, -0.7557],
        [ 0.5862, -0.6619],
        [ 0.5790, -0.6945],
        [ 0.4679, -0.5109]])
"""
```

Right now everything is going well, but it is very inefficient. The trick is very efficient about doing this using matrix multiplication. Considering that the row of a will dot product the column of b and get the fisrt point of c.

```python
torch.manual_seed(42)
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
```

Then let's introduce tril. Some values are ignored at zero position. The reason for it is a sum calcultion is that value of a is one.

```python
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

"""
a=
tensor([[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])
----
b=
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
----
c=
tensor([[ 2.,  7.],
        [ 8., 11.],
        [14., 16.]])
"""
```

If we normalize a, we could get average value in c.

```python
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
"""
a=
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
----
b=
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
----
c=
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])
"""
```

So, with these basis, let's back to the xbow and make it much more efficient using what we've learned. It could get the same value with the version 1. Thinking that @, we matmal triangle matrix (T, T) with (T, C), so we actually calculate the average sum on the column of (T, C).

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (T, T) @ (B, T, C) -> (B, T, C)
torch.allclose(xbow, xbow2)
```

Now, lets using softmax instead of dividing $wei.sum$ as version 3.

```python
import torch.nn.functional as F
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf')) 
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
"""
wei
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
"""
```

todo: version4 self-attention!

## Zero-shot

This is prompt!

![in-context%20learning.png](.images/in-context%20learning.png)

GPT2.0 compared with SOTA zero-shot approaches in many NLP tasks like below. Actually it is not the best one. But you can find out the scaling laws. As the model size increase, you are getting better and better at downstream metrics.

![zero-shot performance.png](.images/zero-shot%20performance.png)

## Dataset

They made a new dataset of millongs of webpages called WebText. 

## FlashAttention

I used FlashAttention to implement GPT2. The note in [LLM-Learning/FlashAttention/README.md](../FlashAttention/README.md)

## Reference

- [GPT2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [GPT，GPT-2，GPT-3 论文精读【论文精读】by 李沐](https://www.bilibili.com/video/BV1AF411b7xQ/?share_source=copy_web&vd_source=3157022a9ba8a59e9a2cac56650df970)
- [Let's build GPT: from scratch, in code, spelled out. by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=hH3vDAtZIzg9pd7-)
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [台大資訊 深度學習之應用 | ADL 14.1: Model Pre-Training 預訓練模型介紹 (GPT, GPT-2) by 
陳縕儂](https://youtu.be/ZQ9b-1ZAT8M?si=_u80sLj9Szb2qU_z)
