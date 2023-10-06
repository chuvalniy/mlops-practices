import json
import os

import click
import mlflow
import torch
import torch.nn as nn
from dotenv import load_dotenv

from model import Transformer

load_dotenv()

# Get remote server credentials from .env and set tracking URI for mlflow.
remote_server_ip = os.getenv("MLFLOW_TRACKING_IP")
remote_server_port = os.getenv("MLFLOW_TRACKING_PORT")
remote_server_uri = f"http://{remote_server_ip}:{remote_server_port}"

mlflow.set_tracking_uri(remote_server_uri)

# Update S3 endpoint URL with current IP address (default is localhost).
s3_server_port = os.getenv("MLFLOW_S3_ENDPOINT_PORT")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{remote_server_ip}:{s3_server_port}"


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
