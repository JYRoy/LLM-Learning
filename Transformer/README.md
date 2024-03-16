# Transformer

- [Transformer](#transformer)
  - [Architecture](#architecture)
  - [Input](#input)
  - [Encoder](#encoder)
    - [Self-Attention](#self-attention)
    - [Multi-head Attention](#multi-head-attention)
  - [Decoder](#decoder)
  - [Output](#output)
  - [Resources](#resources)


## Architecture

![Transformer Architecture.PNG](.images/Transformer%20Architecture.PNG)

## Input

Token Embedding + Segment Embedding + Position Embedding

- Token Embedding: nn.Embedding
- Position Encoding: sin/cos position encoding
  - ![sin_cos_formula.PNG](.images/sin_cos_formula.PNG)
    - pos: absolute position index
    - i: from 0 to embedding length, like (0, 512) if we want 512 embedding length for the token
    - 2i: odd position in embedding, this value is calculated from sin
    - 2i+1: even position in embedding, this value is calculated from cos
  - this way ensures different value for each token between 1 and -1
  - 10000 ** (2 * i / d_model): 2i ranges (0, d_model, 2step), means the frequency downs from 1 to 1/10000
  - ![position_encoding.jpg](.images/position_encoding.jpg)
    - This figure is an real example of position encoding for 60 words(rows) with an embedding size of 512(columns). The sin and cos signals are interweaving.

## Encoder

![Encoder.PNG](.images/Encoder.PNG)

- N encoders
- tow sub-layers in each encoder
- first sub-layer consists of a multi-head attention sub-layer and normalization layer with a residual connection
- second sub-layer consists of a feed forward layer and normalization layer with a residual connection

### Self-Attention

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V
$$

### Multi-head Attention

## Decoder

## Output

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- [Positional Encoding 的高效实现方式](https://zhuanlan.zhihu.com/p/659897051)