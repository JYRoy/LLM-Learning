import torch
from torch import nn
import torch.nn.functional as f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointEmbedding(nn.Module):
    def __init__(self, vocab_size, size):
        super(JointEmbedding, self).__init__()

        self.size = size  # embedding size
        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(vocab_size, size)
        self.positional_emb = nn.Embedding(vocab_size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        """
        input_tensor: dimension (batch_size, seq_len)
        """
        sentence_size = input_tensor.size(-1)
        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, sentence_size // 2 + 1 :] = 1
        pos_tensor = self.numeric_position(sentence_size, input_tensor)
        output = (
            self.token_emb(input_tensor)
            + self.segment_emb(segment_tensor)
            + self.positional_emb(pos_tensor)
        )
        return output

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)


class AttentionHead(nn.Module):

    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp
        self.dim_out = dim_out

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor, attention_mask):
        query, key, value = (
            self.q(input_tensor),
            self.k(input_tensor),
            self.v(input_tensor),
        )

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        scores = scores.masked_fill_(attention_mask, -1e9)

        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)
        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(dim_inp, dim_out) for _ in range(num_heads)]
        )

        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor, attention_mask):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4, droput=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(
            attention_heads, dim_inp, dim_out
        )  # output (batch_size, sentence_size, dim_input)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(droput),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(droput),
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor, attention_mask):
        context = self.attention(input_tensor, attention_mask)
        result = self.feed_forward(context)
        return self.norm(result)


class BERT(nn.Module):
    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads):
        super(BERT, self).__init__()

        self.embedding = JointEmbedding(vocab_size, dim_inp)
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)

        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classification_layer = nn.Linear(dim_inp, 2)

    def forward(self, input_tensor, attention_mask):
        embedded = self.embedding(input_tensor)
        encoded = self.encoder(embedded, attention_mask)

        token_predictions = self.token_prediction_layer(encoded)
        first_word = encoded[:, 0, :]
        return self.softmax(token_predictions), self.classification_layer(first_word)
