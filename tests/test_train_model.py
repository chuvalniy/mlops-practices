import json
import pathlib
import secrets
from collections import OrderedDict

import pytest
import tiktoken
import torch
from click.testing import CliRunner

from src import train_model


@pytest.fixture()
def mock_config(tmp_path: pathlib.Path):
    """
    Creates config file for the model and saves into .json file.
    :param tmp_path: Path to save config file.
    :return:
    """
    temp_dir = tmp_path / "config"
    temp_dir.mkdir()

    encoder = tiktoken.encoding_for_model("gpt2")

    config = {
        "d_model": 1,
        "lr": 1,
        "weight_decay": 1,
        "block_size": 2,
        "n_head": 1,
        "n_layer": 1,
        "dropout": 0.1,
        "batch_size": 2,
        "vocab_size": encoder.n_vocab
    }

    config_path = temp_dir / 'model_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)

    return str(config_path)


@pytest.fixture()
def mock_data(tmp_path: pathlib.Path):
    """
    Creates data tokens from randomly generated hash and saves it into the the file.
    :param tmp_path: Path to save data file.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    data = secrets.token_hex(nbytes=16)  # 16-bit hash to use it like a mock data.

    encoder = tiktoken.encoding_for_model('gpt2')

    tokens = encoder.encode(data)
    tokens_formatted = " ".join(str(token) for token in tokens)

    data_path = temp_dir / 'data.txt'
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(tokens_formatted)

    return str(data_path)


def test_train_model(mock_config: str, mock_data: str, tmp_path: pathlib.Path):
    """
    Tests if model is actually working in terms of end-to-end training.
    :param mock_config: Path to get model config.
    :param mock_data:  Path to get training data.
    :param tmp_path: Path to save model weights.
    :return:
    """

    # Training config.
    train_iter = 100
    val_iter = 10
    split_ratio = 0.5  # Be aware of setting it too low or torch.randint will have an error of wrong boundaries.

    output_path = tmp_path / "weights.pt"

    # train_model() function uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(
        train_model,
        [mock_data, mock_config, str(output_path), str(train_iter), str(val_iter), str(split_ratio)]
    )

    assert result.exit_code == 0

    # Test if weights.pt file exists.
    weights = torch.load(output_path)

    assert isinstance(weights, OrderedDict)
