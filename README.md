# LLM-Learning

Learning how to build and fine-tune some large language models from scratch inpsired by Adrej Karpathy's course [《Let's build GPT: from scratch, in code, spelled out.》](https://youtu.be/kCc8FmEb1nY?si=7hjbzwdxNODVoK_v)

## Catalog

- [LLM-Learning](#llm-learning)
  - [Catalog](#catalog)
  - [Supported](#supported)
  - [Awesome LLM](#awesome-llm)
    - [Model](#model)
    - [Fine-tuning](#fine-tuning)
  - [Environments](#environments)


## Supported

Here are the models and algorithms I implemented and noted.

- Bigram
- Transformer

## Awesome LLM

Here are some milestone works in LLM, but I'll just pick some of them and implement simplified version

### Model

|  Date   |   keywords   | Institute | Paper|
| :-----: | :----------: | :-------: | :--: |
| 2017-06 | Transformers |  Google   | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)|
| 2018-06 |   GPT 1.0    |  OpenAI   | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)|
| 2018-10 |     BERT     |  Google   | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)|
| 2019-02 |   GPT 2.0    |  OpenAI   | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)|
| 2020-05 |   GPT 3.0    |  OpenAI   | [Language models are few-shot learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)|
| 2023-02 |    LLaMA     |   Meta    | [LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)|
| 2023-03 |    GPT 4     |  OpenAI   | [GPT-4 Technical Report](https://openai.com/research/gpt-4)|
| 2023-05 |     RWKV     |  Bo Peng  | [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)|
| 2023-07 |   LLaMA 2    |   Meta    | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)|
| 2023-10 |  Mistral 7B  |  Mistral  | [Mistral 7B](https://arxiv.org/pdf/2310.06825.pdf%5D%5D%3E)|
| ... |  ...   |  ...   | ...  |


### Fine-tuning

|  Date   |   keywords   | Institute | Paper|
| :-----: | :----------: | :-------: | :--: |
| 2021-05 | LoRA |  Microsoft   | [LORA: LOW-RANK ADAPTATION OF LARGE LAN-GUAGE MODELS](https://arxiv.org/abs/2106.09685)|
| ... |  ...   |  ...   | ...  |

## Environments

- Ubuntu 22.04
- Nvidia RTX 3080
- Intel i7-11700K
- torch 2.2.0+cu121
