import json

import click
import tiktoken


@click.command()
@click.argument("d_model", type=click.INT)
@click.argument("lr", type=click.FLOAT)
@click.argument("weight_decay", type=click.FLOAT)
@click.argument("block_size", type=click.INT)
@click.argument("n_head", type=click.INT)
@click.argument("n_layer", type=click.INT)
@click.argument("dropout", type=click.FLOAT)
@click.argument("batch_size", type=click.INT)
@click.argument("output_path", type=click.Path())
def create_config(
        d_model: int,
        lr: float,
        weight_decay: float,
        block_size: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        batch_size: int,
        output_path: str
):
    """
    Creates config for the transformer model in a JSON format.
    Also adds vocab_size to the output JSON file.
    :param d_model: Model dimension.
    :param lr: Learning rate.
    :param weight_decay: Regularization coefficient.
    :param block_size: Length of each token sequence.
    :param n_head: Number of attention heads.
    :param n_layer: Number of attention blocks.
    :param dropout: Dropout probability.
    :param batch_size: Number of examples to process at once.
    :param output_path: Path to the model config file.
    :return:
    """

    # Get vocab_size from tiktoken encoder.
    encoder = tiktoken.encoding_for_model("gpt2")
    vocab_size = encoder.n_vocab  # Need this in transformer model.

    config = {
        "d_model": d_model,
        "lr": lr,
        "weight_decay": weight_decay,
        "block_size": block_size,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "batch_size": batch_size,
        "vocab_size": vocab_size
    }

    # Save config into JSON file.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)


if __name__ == "__main__":
    create_config()
