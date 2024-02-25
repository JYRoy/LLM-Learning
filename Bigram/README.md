# Bigram

- [Bigram](#bigram)
  - [What is bigram model?](#what-is-bigram-model)
  - [Concepts](#concepts)
  - [Resources](#resources)

## What is bigram model?

the conditional probability of the preceding word give the previous one `P(w_n|w_{n−1})` 

the core is that the model will predict the next token by the previous token, the embedding vector is the probilities.

for example:

- start of content -> h
- h -> i
- i -> i
- i -> ' '
- ' ' -> t
- t -> h
- h -> r
- r -> e
- r -> e

## Concepts

- Block Size
  - the token sequence size, the number of tokens taken from the data as the input sequence.
  - the entire block is typically fed into the model as a sequence for prediction as above
    - in the actual train process, the block will be inputted as one and predict the next tokens of each iuput in the block, like below
    - input tokens([:block_size]): [56,  1, 44, 39, 58, 46, 43, 56] 10
    - output tokens([1:block_size+1]): 56 [ 1, 44, 39, 58, 46, 43, 56, 10]

- nn.Embedding
  - vocab lookup table
  - input: vocab index, (batch_size, block_size) like (32, 8)
  - output: embedding tensor, (batch_size, block_size, embedding_dim) like (32, 8, 64)
  - the embedding tensor is used to represent the input tokens in a continuous vector space and it also could be used as the prediction scores

-  Train process of the bigram model
   - using 256 tokens(32, 8) as input to predict the next 256(32, 8) tokens
   - the model generates 256(32, 8) predicted tokens as logits, actually it's (32, 8, 64) which is (batch_size, block_size, embedding_dim)
   - The output logits are passed through a softmax layer to obtain probability distributions over the vocabulary
   - model receives 256(32, 8) tokens as targets
   - targets(32, 8) and logits (32, 8, 64) will be used to compute [cross_entropy_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) which input shape is (N, C) and target shape is (N)
    - N: batch size
    - C: number of classes

## Resources

- [Standord Course: N-gram](extension://oikmahiipjniocckomdccmplodldodja/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fweb.stanford.edu%2F~jurafsky%2Fslp3%2F3.pdf)