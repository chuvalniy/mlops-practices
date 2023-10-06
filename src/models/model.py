import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention.
    For the details visit https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model: int, block_size: int, n_head: int, dropout: float = 0.1):
        """
        :param d_model: Model dimension.
        :param block_size: Length of each token sequence.
        :param n_head: Number of attention heads.
        :param dropout: Dropout probability.
        """
        super().__init__()

        self.head_dim = d_model // n_head
        self.n_head = n_head

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Since we're using decoder-only transformer, we need to define the mask.
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: Tensor):
        """
        Performs multi-head attention with query, key and value reshaping.

        :param x: Input tensor of (N, T, E) dimension.
        :return: Tensor with (N, T, E) dimension.
        """
        N, T, E = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape into (N, T, n_head, head_dim) -> Transpose into (N, n_head, T, head_dim)
        q = q.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(N, T, self.n_head, self.head_dim).transpose(1, 2)

        wei = q @ k.transpose(-1, -2) / self.head_dim ** 0.5  # Scaled dot-product
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = wei @ v

        # Transpose & reshape into original shape.
        wei = wei.transpose(1, 2).reshape(N, T, E)
        wei = self.proj(wei)

        return self.dropout(wei)


class FeedForward(nn.Module):
    """
    Feed-forward module of transformer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        :param d_model: Model dimension.
        :param dropout: Dropout probability.
        """
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        """
        Execute feed-forward module using nn.Sequential()

        :param x: Input tensor of (N, T, E) dimension.
        :return: Output tensor of (N, T, E) dimension.
        """
        return self.ff(x)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for transformer model
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        :param d_model: Model dimension.
        :param dropout: Dropout probability.
        :param max_length: Maximum length of the input sequence.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        """
        Applies positional encoding to input tensor.

        :param x: Input tensor of (N, T, E) dimension.
        :return: Output tensor of (N, T, E) dimension.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder (Multi-head attention & Feed-forward).
    """

    def __init__(self, d_model: int, block_size: int, n_head: int, dropout: float = 0.1):
        """
        :param d_model: Model dimension.
        :param block_size: Length of each token sequence.
        :param n_head: Number of attention heads.
        :param dropout: Dropout probability.
        """
        super().__init__()

        self.layer_norm_mha = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, block_size, n_head, dropout)

        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: Tensor):
        """
        Calculates multi-head attention (MHA) and feed-forward (FF).
        Applies layer normalization and skip-connections to MHA and FF outputs.
        :param x: Input tensor of shape (N, T, E).
        :return: Output tensor of shape (N, T, E).
        """
        x = x + self.layer_norm_mha(self.mha(x))
        x = x + self.layer_norm_ff(self.ff(x))

        return x


class Transformer(nn.Module):
    """
    Decoder-only transformer.
    """

    def __init__(self, d_model: int, vocab_size: int, block_size: int, n_head: int, n_layer: int, dropout: float = 0.1):
        """
        :param d_model: Model dimension.
        :param vocab_size: Size of the text corpus vocabulary.
        :param block_size: Length of each token sequence.
        :param n_head: Number of attention heads.
        :param n_layer: Number of attention blocks.
        :param dropout: Dropout probability.

        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.block_size = block_size

        # Preprocess tokens before putting inputs to the transformer.
        self.tokens_emb = nn.Embedding(vocab_size, d_model)
        self.tokens_pe = PositionalEncoding(d_model, dropout=dropout)

        # Transformer block.
        self.transformer = nn.Sequential(
            *[TransformerDecoder(d_model, block_size, n_head, dropout) for _ in range(n_layer)]
        )

        # Linear layer for projection purposes.
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor):
        """
        Forward pass through the whole model.

        :param x: Input tensor of shape (N, T).
        :return: Output tensor of shape (N, T, vocab_size).
        """
        x = self.tokens_emb(x)
        x = self.tokens_pe(x)

        x = self.transformer(x)
        x = self.fc(x)

        return x
