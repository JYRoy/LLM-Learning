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

Before starting, I want to highlight that the following content is basicing on Karypthy's video: [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=ukAFXFUoX3i1binA)

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

version4 self-attention!

The problem self-attention solves is gathering information from the past and want to do it in a data dependent way. 

The way self-attention solves this problem is that every sinle node or every sinle token at each position will emit two vectors: query and key. The query vector could see as what this token is looking for and the key vector is what do this token contain. 

The way we get the affinities between these tokens now in a sequence is we bsically just do a dot product between the keys and the queries. The current tokens' query dot products with all previous tokens' keys of all the other tokens and that dot product becomes weight, but actually we name it attention score. So, if the key and query are sort of aligned, they will interact to a very high amount. And then I will get to learn more about the specific token as opposed to any other token in the sequence.

Let's start coding!

I'm just going to show you the whole single head attention code and comment it in detail.

```python
B, T, C = 4, 8, 32  # batch size, sequence length(times), channels(embedding size)
x = torch.randn(B, T, C)

head_size = 16  # a single head size, in the multi head attention version there are serval heads concated together
key = nn.Linear(C, head_size, bias=False)  # use a key matrix get a key, the key length is head size
query = nn.Linear(C, head_size, bias=False)  # use a query matrix get a query, the query length is head size
value = nn.Linear(C, head_size, bias=False)  # use a value matrix get a value, the value length is head size
k = key(x)  # (B, T, head_size) -> (B, T, 16), for each token, the key is a 16 length vector
q = query(x)  # (B, T, head_size) -> (B, T, 16), for each token, the query is a 16 length vector
wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))  # mask matrix, it a lower triangular matrix
wei = wei.masked_fill(tril==0, float('-inf'))  # the the upper triangular part as -inf
wei = F.softmax(wei, dim=-1)  # after softmax the -inf value becomes 0, the lower part becomes attention score(weight) with each other
v = value(x)
out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C): torch.Size([4, 8, 32])
```

The wei is

```shell
tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
         [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
         [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
         [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1687, 0.8313, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2477, 0.0514, 0.7008, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4410, 0.0957, 0.3747, 0.0887, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0069, 0.0456, 0.0300, 0.7748, 0.1427, 0.0000, 0.0000, 0.0000],
         [0.0660, 0.0892, 0.0413, 0.6316, 0.1649, 0.0069, 0.0000, 0.0000],
         [0.0396, 0.2288, 0.0090, 0.2000, 0.2061, 0.1949, 0.1217, 0.0000],
         [0.3650, 0.0474, 0.0767, 0.0293, 0.3084, 0.0784, 0.0455, 0.0493]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4820, 0.5180, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1705, 0.4550, 0.3745, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0074, 0.7444, 0.0477, 0.2005, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.8359, 0.0416, 0.0525, 0.0580, 0.0119, 0.0000, 0.0000, 0.0000],
         [0.1195, 0.2061, 0.1019, 0.1153, 0.1814, 0.2758, 0.0000, 0.0000],
         [0.0065, 0.0589, 0.0372, 0.3063, 0.1325, 0.3209, 0.1378, 0.0000],
         [0.1416, 0.1519, 0.0384, 0.1643, 0.1207, 0.1254, 0.0169, 0.2408]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.6369, 0.3631, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2586, 0.7376, 0.0038, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4692, 0.3440, 0.1237, 0.0631, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1865, 0.4680, 0.0353, 0.1854, 0.1248, 0.0000, 0.0000, 0.0000],
         [0.0828, 0.7479, 0.0017, 0.0735, 0.0712, 0.0228, 0.0000, 0.0000],
         [0.0522, 0.0517, 0.0961, 0.0375, 0.1024, 0.5730, 0.0872, 0.0000],
         [0.0306, 0.2728, 0.0333, 0.1409, 0.1414, 0.0582, 0.0825, 0.2402]]],
       grad_fn=<SoftmaxBackward0>)
```

The meaning of the wei matrix is that the element on the main diagonal is the current token and the past values are the attention scores of these tokens with the current token. Take the fist (T, T) matrix's last row as example. The fourth value is a bigger one, it means that the fourth token has a high affinity with the eight token and the seventh token yet.

```shell
[0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]
```

In the end, it is aggregating the value which is the thing that gets aggregated for the purposes of this single head between the different nodes. We also use a value matrix get the value for each tokens.

```python
v = value(x)
out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C): torch.Size([4, 8, 32])
```

Notes:

- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information via a weighted sum from all of the nodes that points to it.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other. @ we used above is actually a batched matrix multiply that applies basically a matrix multiplication kind of in parallel across the batch dimension.
- In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. In an "decoder" attention block, it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- What is the difference between self-attention, cross-attention?
  - self-attention just means that the keys and values are produced from the same source as queries.
  - cross-attention means the queries are produced from x, but the keys and values come from some other, external source(e.g. an encoder module)
- "Scaled" attention additional divides attention score by 1/sqrt(head_size). I think there are tow reasons
  - Prevents the input value for softmax being too large. If it is too large, the jacobian matrix will close to zero matrix and then the gradients will also close to zero. (It is related to the backpropagation of softmax, needs a indepth analysis)
  - This makes it so when input Q, K are unit variance, attention score will be unit variance too and Softmax will stay diffuse and not saturate too much, which means that updation will become more stable. The illustration below

```python
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1)
k.var()  # tensor(0.9487)
q.var()  # tensor(1.0449)
wei.var()  # tensor(14.3682)
```

We could see that if we do not use 1/sqrt(head_siz). The variance of wei is close to the head size.

```python
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1)
k.var()  # tensor(1.0700)
q.var()  # tensor(0.9006)
wei.var()  # tensor(1.1277)
```

If we scaled attention, the variance of wei is close to one.


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
- [保姆级分析self Attention为何除根号d，看不懂算我的](https://zhuanlan.zhihu.com/p/503321685)
