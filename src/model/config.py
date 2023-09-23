import torch
from src.model.encoder import encoder


device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = encoder.n_vocab

block_size = 256

n_layers = 12
n_head = 6
d_model = 384

dropout = 0.2