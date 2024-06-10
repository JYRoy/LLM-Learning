# GPT1.0

- [GPT1.0](#gpt10)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Framework](#framework)
    - [Unsupervised pre-training](#unsupervised-pre-training)

## Overview

Title is "Improving Language Understanding by Generative Pre-Training" means that tasks can be realized by **generative pre-training** of a **language model** on a diverse corpus of **unlabeled text**, followed by discriminative fine-tuning on each specific task. This approach called **semi-supervised** approach.

## Architecture

The model architecture is basing on Transformer decoder instead of RNN cause Transformer coudl handle long-term context and structured memory, and task-specific input adaptations.

## Framework

### Unsupervised pre-training

GPT use a standard language modeling objective to maximize the following likelihood means the using k words (words in window size) to predict the next word $u_i$

$$
L_1(U) = \sum_i logP(u_i|u_{i-k},...,u_{i-1};\vartheta)
$$

- k: size of the context window
- P: the conditional probability P is modeled using nerual network with parameters $\vartheta$
- $\vartheta$: model parameters
