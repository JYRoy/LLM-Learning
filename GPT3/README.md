# GPT3

- [GPT3](#gpt3)
  - [Overview](#overview)
  - [In-context Learning](#in-context-learning)
  - [Architecture](#architecture)
  - [Training](#training)
    - [Batch size and Learning Rate](#batch-size-and-learning-rate)
    - [Hypre-parameters](#hypre-parameters)
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

GPT3 has 8 different model sizes. 

![GPT3 model size.png](./.images/GPT3%20model%20size.png)

## Training

### Batch size and Learning Rate

It is confused that larger model size and larger batch size using smaller learning rate. Author reached this conclusion on [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) and [An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162). But, I cannot got the same conclusion. It looks like that these paper cannot get these conclusion and many people think that the larger batchsize with larger learning rate. 

According to these papers, we could get the following conculsions:

1. Performance depends strongly on scale, weakly on model shape: scale includes model parameters(N), dataset(D) and calculation amount(C), shape includes model depth, model width and self-attention heads etc.
2. Smooth power laws: performance has a power-law relationship with each of the three scale factors N, D and C when not bottlenecked by the other two.

![power-law relationship.png](.images/power-law%20relationship.png)

3. Universality of overfitting: Performance improves predictably as long as we scale up N and D in tandem, but enters a regime of diminishing returns if either N or D is held fixed while the other increases. The performance penalty depends predictably on the ratio ${N^{0.74}}/{D}$, meaning that every time we increase the model size 8x, we only need to increase the data by roughly 5x to avoid a penalty.
4. Universality of training: Training curves follow predictable power-laws whose parameters are roughly independent of the model size. By extrapolating the early part of a training curve, we can roughly predict the loss that would be achieved if we trained for much longer.
5. Transfer improves with test performance: transfer to a different distribution incurs a constant penalty but otherwise improves roughly in line with performance on the training set.
6. Sample efficiency: Large models are more sample-efficient than small models, reaching the same level of performance with fewer optimization steps and using fewer data point.
7. Convergence is inefficient:When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models and stopping significantly short of convergence.
8. Optimal batch size: The ideal batch size for training these models is roughly a power of the loss only, and continues to be determinable by measuring the gradient noise scale; it is roughly 1-2 million tokens at convergence for the largest models we can train.
9. Learning rate tuning: the choice of learning rate schedule is mostly irrelevant, as long as the total summed learning rate is sufficiently large, and the schedule includes a warmup period and a final decay to near-vanishing learning rate. As long as the learning rate is not too small and does not decay too quickly, performance does not depend strongly on learning rate.

### Hypre-parameters

- Adam: $\beta_1=0.9$, $\beta_2=0.95$ and $\epsilon=10^{-8}$
- Clip the global norm of the gradient at 1.0
- Cosine decay for learning rate down to 10% of its value over 260 billion tokens(after 260 billion tokens, training continues at 10% of the original learning rate)
- Linear LR warmup over the first 375 million tokens.
- Gradually increase the batch size linearly from a small value(32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size.
- Data are sampled without replacement during training to minimize overfitting.
- All models use weight decay of 0.1 to provide a small amount of reguarlization.
- Train on sequences of the full 2048 toekn context window.
- Delimit with a special end of text token to separate the unrelated text tokens.
- 

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
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
