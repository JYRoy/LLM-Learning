# GPT3

- [GPT3](#gpt3)
  - [Overview](#overview)
  - [In-context Learning](#in-context-learning)
  - [Architecture](#architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Result](#result)
  - [Reference](#reference)


## Overview

Language Models are Few-Shot Learners.

GPT 3 is an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model. (sparse means that model has a lot of zeros in its parameters).

GPT3 is also applied without any gradient updates or fine-tuning, instead using"in-context learning".

## In-context Learning

The way i see it, the reasons in-context learning has good performance are that larger model size and data diversity.

![in-context%20learning.png](../GPT2/.images/in-context%20learning.png)


## Architecture

GPT3 has same model and architecture as GPT2 including modified initialization, pre-normalization and reversible tokenization.

GPT3 has 8 different model sizes. (It is confused that larger model size and larger batch size using smaller learning rate.).

![GPT3 model size.png](./.images/GPT3%20model%20size.png)

## Training

todo

## Evaluation

They evaluate each example in the evaluation set by randomly drawing K examples from that task's training set as conditioning.

For most tasks they compare the per-token likelihood and the prompt is "Answer: " or  "A: ".

On tasks that involve binary classification, they used "True" or "False" rather than "0" or "1".

On tasks with free-form completion like questiong answering, they used beam search with a beam width of 4 and a length penalty of $\alpha = 0.6$

## Result

Below shows the the performance of the model increases linearly with the exponential increase of computation. This is very helpful in predicting the model convergence.

![smooth scaling of performance with compute.png](.images/smooth%20scaling%20of%20performance%20with%20compute.png)

Few-shot shows a strong boost to accuracy. It also shows scaling character.

![few-shot capability.png](.images/few-shot%20capability.png)

## Reference

- [GPT3 Paper](https://arxiv.org/pdf/2005.14165)
- [GPT，GPT-2，GPT-3 论文精读【论文精读】by 李沐](https://www.bilibili.com/video/BV1AF411b7xQ/?share_source=copy_web&vd_source=3157022a9ba8a59e9a2cac56650df970)
- [Let's build GPT: from scratch, in code, spelled out. by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=hH3vDAtZIzg9pd7-)
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
