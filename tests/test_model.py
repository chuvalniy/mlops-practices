import pytest
import torch

from src import Transformer


@pytest.fixture()
def transformer_model():
    """
    Fixture to initialize model and use it in further tests.

    :return: Transformer model.
    """
    d_model = 16
    n_head = 4
    n_layer = 2

    block_size = 8
    vocab_size = 32

    model = Transformer(d_model=d_model, vocab_size=vocab_size, block_size=block_size, n_head=n_head, n_layer=n_layer)

    return model


def test_model(transformer_model: Transformer):
    """
    Tests the output dimensions & type of the model output.

    :param transformer_model: Transformer model.
    :return: None
    """

    block_size = transformer_model.block_size
    vocab_size = transformer_model.vocab_size

    example_data = torch.ones(size=(1, block_size), dtype=torch.int64)

    output = transformer_model(example_data)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, block_size, vocab_size)
