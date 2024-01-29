import pathlib

import pandas as pd
import numpy as np
import pytest
from click.testing import CliRunner

from src.models.prepare_datasets import prepare_datasets


@pytest.fixture()
def mock_csv_file(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.csv file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    test_array = np.zeros(shape=(10, 10))
    test_df = pd.DataFrame(test_array)

    test_csv_path = temp_dir / 'mock.csv'
    test_df.to_csv(test_csv_path, index=False)

    return str(test_csv_path)


def test_prepare_datasets(mock_csv_file, tmp_path: pathlib.Path):
    """
    Checks whether the data has been split correctly.
    :param mock_csv_file: File path created in mock_csv_file().
    :param tmp_path: Temporary file path provided via pytest API.
    :return: None
    """
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"

    # clean_data() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(prepare_datasets, [mock_csv_file, str(train_path), str(val_path)])

    assert result.exit_code == 0

    # Expected shapes after 30% split for validation set.
    expected_train_shape = (7, 10)
    expected_val_shape = (3, 10)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    assert expected_train_shape == train_df.shape
    assert expected_val_shape == val_df.shape
