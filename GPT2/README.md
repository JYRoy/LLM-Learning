# GPT2

- [GPT2](#gpt2)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Zero-shot](#zero-shot)
  - [Dataset](#dataset)
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

## Zero-shot

This is prompt!

![in-context%20learning.png](.images/in-context%20learning.png)

GPT2.0 compared with SOTA zero-shot approaches in many NLP tasks like below. Actually it is not the best one. But you can find out the scaling laws. As the model size increase, you are getting better and better at downstream metrics.

![zero-shot performance.png](.images/zero-shot%20performance.png)

## Dataset

They made a new dataset of millongs of webpages called WebText. 

## Reference

- [GPT2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [GPT，GPT-2，GPT-3 论文精读【论文精读】by 李沐](https://www.bilibili.com/video/BV1AF411b7xQ/?share_source=copy_web&vd_source=3157022a9ba8a59e9a2cac56650df970)
- [Let's build GPT: from scratch, in code, spelled out. by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=hH3vDAtZIzg9pd7-)
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [台大資訊 深度學習之應用 | ADL 14.1: Model Pre-Training 預訓練模型介紹 (GPT, GPT-2) by 
陳縕儂](https://youtu.be/ZQ9b-1ZAT8M?si=_u80sLj9Szb2qU_z)
