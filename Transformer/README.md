# Transformer

- [Transformer](#transformer)
  - [Architecture](#architecture)
  - [Input](#input)
  - [Output](#output)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
  - [Resources](#resources)


## Architecture

![Transformer Architecture.PNG](.images/Transformer%20Architecture.PNG)

## Input

Token Embedding + Segment Embedding + Position Embedding

- Token Embedding: nn.Embedding
- Position Encoding: sin/cos position encoding
  - ![sin_cos_formula.PNG](.images/sin_cos_formula.PNG)
    - pos: absolute position index
    - 2i: odd position in embedding
    - 2i+1: even position in embedding
  - this way ensures different value for each token
  - 10000 ** (2 * i / d_model): 2i ranges (0, d_model, 2step), means the frequency downs from 1 to 1/10000

## Output

## Encoder

## Decoder

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- [Positional Encoding 的高效实现方式](https://zhuanlan.zhihu.com/p/659897051)