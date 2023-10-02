import torch

from src import Transformer


def test_model_dimensions():
    d_model = 16
    n_head = 4
    n_layer = 2

    batch_size = 4
    block_size = 8
    vocab_size = 32

    model = Transformer(d_model=d_model, vocab_size=vocab_size, block_size=block_size, n_head=n_head, n_layer=n_layer)

    example_data = torch.ones(size=(batch_size, block_size), dtype=torch.int64)

    output = model(example_data)

    expected_output = torch.ones(size=(batch_size, block_size, vocab_size), dtype=torch.int64)

    assert output.shape == expected_output.shape
