import pathlib

import pytest
import tiktoken
from click.testing import CliRunner

from src.data.tokenize_data import tokenize_data


@pytest.fixture()
def mock_txt_path(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.txt file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    txt_data = "hello world"

    txt_file_path = temp_dir / "mock.txt"
    with open(txt_file_path, "w", encoding='utf-8') as f:
        f.write(txt_data)

    return str(txt_file_path)


def test_tokenize_data(mock_txt_path: str, tmp_path: pathlib.Path):
    """
    Checks whether text was actually tokenized.
    :param mock_txt_path: File path created in mock_txt_path().
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    output_path = tmp_path / "tokens.txt"

    # tokenize_data() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(tokenize_data, [mock_txt_path, str(output_path)])

    assert result.exit_code == 0

    with open(output_path, 'r', encoding='utf-8') as f:
        tokens_str = f.read()

    # Check whether the contents of the list can be interpreted as numbers.
    tokens_str_list = tokens_str.split()
    assert all(x.isdigit() for x in tokens_str_list)

    tokens = [int(c) for c in tokens_str_list]

    encoder = tiktoken.get_encoding("gpt2")  # Same as in tokenize_data()
    text_to_encode = "hello world"  # Same as txt_data in mock_txt_path()

    expected_tokens = encoder.encode(text_to_encode)

    assert all(x == y for x, y in zip(tokens, expected_tokens))
