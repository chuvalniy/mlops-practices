import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

__all__ = ['Transformer']


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, block_size: int, n_head: int, dropout: float = 0.1):
        super().__init__()

        self.head_dim = d_model // n_head
        self.n_head = n_head

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: Tensor):
        N, T, E = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(N, T, self.n_head, self.head_dim).transpose(1, 2)

        wei = q @ k.transpose(-1, -2) / self.head_dim ** 0.5

        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = wei @ v

        wei = wei.transpose(1, 2).reshape(N, T, E)
        wei = self.proj(wei)

        return self.dropout(wei)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        return self.ff(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, block_size: int, n_head: int, dropout: float = 0.1):
        super().__init__()

        self.layer_norm_mha = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, block_size, n_head, dropout)

        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: Tensor):
        x = x + self.layer_norm_mha(self.mha(x))
        x = x + self.layer_norm_ff(self.ff(x))

        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, block_size: int, n_head: int, n_layer: int, dropout: float = 0.1):
        super().__init__()
        
        self.tokens_emb = nn.Embedding(vocab_size, d_model)
        self.tokens_pe = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Sequential(
            *[TransformerDecoder(d_model, block_size, n_head, dropout) for _ in range(n_layer)]
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor):
        x = self.tokens_emb(x)
        x = self.tokens_pe(x)

        x = self.transformer(x)
        x = self.fc(x)

        return x
