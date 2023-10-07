import json
import math
import os

import click
import mlflow
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch import Tensor
from torch.nn import functional as F

load_dotenv()

# Get remote server credentials from .env and set tracking URI for mlflow.
remote_server_ip = os.getenv("MLFLOW_TRACKING_IP") if os.getenv("MLFLOW_TRACKING_IP") is not None else "localhost"
remote_server_port = os.getenv("MLFLOW_TRACKING_PORT") if os.getenv("MLFLOW_TRACKING_PORT") is not None else 5000
remote_server_uri = f"http://{remote_server_ip}:{remote_server_port}"

mlflow.set_tracking_uri(remote_server_uri)

# Update S3 endpoint URL with current IP address (default is localhost).
s3_server_port = os.getenv("MLFLOW_S3_ENDPOINT_PORT")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{remote_server_ip}:{s3_server_port}"


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


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.argument("train_iter", type=click.INT)
@click.argument("val_iter", type=click.INT)
@click.argument("split_ratio", type=click.FLOAT)
def train_model(input_path: str, config_path: str, output_path: str, train_iter: int, val_iter: int,
                split_ratio: float):
    """
    Trains transformer model for a number of times and create model weights file in .pt format.
    :param input_path: Path to the data.
    :param config_path: Path to the model config file.
    :param output_path: Path to save model weights in .pt format.
    :param train_iter: Number of training iterations.
    :param val_iter: Number of validation iterations.
    :param split_ratio: Train and validation split ratio.
    :return:
    """

    with mlflow.start_run():
        # Load data and split into train & validation sets.
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.read()

        data = torch.tensor([int(c) for c in data.split()], dtype=torch.long)
        train_data, val_data = _train_test_split(data, split_ratio)

        # Load config file and define a model.
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)

        mlflow.log_params(model_config)

        model = Transformer(
            d_model=model_config['d_model'],
            vocab_size=model_config['vocab_size'],
            block_size=model_config['block_size'],
            n_head=model_config['n_head'],
            n_layer=model_config['n_layer'],
            dropout=model_config['dropout']
        )

        # Define optimizer & criterion for loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['lr'])
        criterion = nn.CrossEntropyLoss()

        # Training process.
        for idx in range(train_iter):
            X, y = _get_batch(train_data, model_config['block_size'], model_config['batch_size'])

            B, T = X.shape

            # Forward pass w/ loss calculation.
            output = model(X)

            output = output.view(B * T, -1)
            y = y.view(B * T)

            train_loss = criterion(output, y)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

            # Validate model
            if idx % val_iter == 0:
                val_loss = _validate_model(model, criterion, val_data, val_iter, model_config)
                print(f"Iteraion {idx}/{train_iter} | Train loss: {train_loss} | Val loss: {val_loss}")

                mlflow.log_metric("loss", val_loss, step=idx)  # Track loss in mlflow.

        mlflow.pytorch.log_model(model, "transformer_baseline")
        torch.save(model.state_dict(), output_path)  # Save model weights.


def _train_test_split(data: torch.Tensor, split_ratio: float):
    """
    Gets a whole dataset and splits it into training and validation sets.
    :param data: Dataset to split.
    :param split_ratio: Ration of validation set.
    :return: Training and validation data (torch.Tensor, torch.Tensor)
    """
    val_split = int(split_ratio * len(data))

    train_data = data[val_split:]
    val_data = data[:val_split]

    return train_data, val_data


def _get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    """
    Randomly gets a single batch of data depending on the block_size and batch_size.
    :param data: Data to obtain the batch.
    :param block_size: Length of a single batch element specified as token sequence.
    :param batch_size: Number of elements in the batch.
    :return: X, y (torch.Tensor, torch.Tensor).
    """

    ix = torch.randint(len(data) - block_size, size=(batch_size,))  # List of ints that serve as sequence start index.

    # Stack each of the token sequence to get single data batch (batch_size, block_size).
    X = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return X, y


@torch.no_grad()
def _validate_model(model: Transformer, criterion: nn.CrossEntropyLoss, data: torch.Tensor, val_iter: int,
                    config: dict[str]):
    """
    Specifies model validation and calculates validation loss.
    :param model: Transformer model.
    :param criterion: Criterion to get the loss.
    :param data: Validation data.
    :param val_iter: Number of validation iterations.
    :param config: Model config.
    :return: Validation loss (torch.float)
    """
    model.eval()  # Turn on evaluation mode.

    running_loss = 0.0
    for i in range(val_iter):
        X, y = _get_batch(data, config["block_size"], config["batch_size"])

        B, T = X.shape

        # Forward pass
        output = model(X)

        output = output.view(B * T, -1)
        y = y.view(B * T)

        loss = criterion(output, y)

        running_loss += loss.item()  # Accumulate validation loss.

    model.train()  # Switch back to training mode.

    val_loss = running_loss / val_iter  # Calculate mean validation loss.
    return val_loss


if __name__ == "__main__":
    train_model()
