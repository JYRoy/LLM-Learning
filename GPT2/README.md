# GPT2

- [GPT2](#gpt2)
  - [Overview](#overview)
  - [Architecture](#architecture)
    - [Attention Module](#attention-module)
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

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # get qkv in one maxtic calculation that concat weight Q, K, V into one weight as linear
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a bias, actually a mask, just following the OpenAI naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension (n_embd)
        # nh is "number of heads", hs is "head size", C is "number of channels" = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # batch size, seq len, n_embd

        # split n_embd into serveral heads so that dim2 means n_heads and dim3 means C(n_embd) // n_heads
        # it means that each head for q/k/v has C // n_heads(hs) embedding dimensions
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
```

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
