import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import src.model.config as cfg
import src.model.encoder as enc


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()

        self.n_head = n_head
        self.head_size = d_model // n_head

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, q, k, v, require_mask=True):
        N, T, E = q.shape

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(N, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(N, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(N, T, self.n_head, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        if require_mask is not None:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        output = F.softmax(wei, dim=-1)
        output = output @ v

        output = output.transpose(1, 2).reshape(N, T, E)
        output = self.proj(output)

        return self.dropout(output)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.ff(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()

        self.l_norm_mha = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head, dropout, block_size)

        self.l_norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, src):
        src = src + self.l_norm_mha(self.mha(src, src, src))
        src = src + self.l_norm_ff(self.ff(src))

        return src


class GenerativeLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size, block_size, n_layers, n_head, dropout):
        super().__init__()
        self.block_size = block_size

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(block_size, d_model)

        self.transformer = nn.Sequential(
            *[TransformerDecoder(d_model, n_head, dropout, block_size) for _ in range(n_layers)]
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        B, T = src.shape

        tok_emb = self.emb(src)
        pos_emb = self.pe(torch.arange(T, device=cfg.device))

        src = tok_emb + pos_emb

        src = self.transformer(src)

        logits = self.fc(src)

        return logits

    def generate(self, src, max_length):
        for _ in range(max_length):
            src_cond = src[:, -self.block_size:]

            logits = self(src_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)
            src = torch.cat((src, idx_next), dim=1)

        return src


def get_model():
    model = GenerativeLanguageModel(
        cfg.d_model,
        cfg.vocab_size,
        cfg.block_size,
        cfg.n_layers,
        cfg.n_head,
        cfg.dropout
    )

    model.load_state_dict(torch.load("src/model/transformer_decoder_gpt_2_tokenizer.pt", map_location=cfg.device))
    model.eval()

    return model


def generate_and_decode_text(model, n_words=500):
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    print(enc.encoder.decode(model.generate(context, n_words)[0].tolist()))
