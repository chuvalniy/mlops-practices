import json
import pathlib

import tiktoken
from click.testing import CliRunner

from src import create_config


def test_create_config(tmp_path: pathlib.Path):
    """
    Checks if created config is the same as expected one.

    :param tmp_path: Path to save config file.
    :return:
    """
    output_path = tmp_path / 'model_config.json'

    d_model = 16
    n_head = 4
    n_layer = 2

    block_size = 8
    batch_size = 4
    dropout = 0.1

    # create_config() function uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(
        create_config,
        [str(d_model), str(block_size), str(n_head), str(n_layer), str(dropout), str(batch_size), str(output_path)]
    )

    # Check if the function passed at all.
    assert result.exit_code == 0

    # Test if model_config.json has the same data as expected_config
    with open(output_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    encoder = tiktoken.encoding_for_model("gpt2")
    vocab_size = encoder.n_vocab

    expected_config = {
        "d_model": d_model,
        "block_size": block_size,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "batch_size": batch_size,
        "vocab_size": vocab_size
    }

    assert all(v_actual == v_expected for v_actual, v_expected in zip(config.values(), expected_config.values()))
