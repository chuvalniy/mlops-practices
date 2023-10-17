import pathlib

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data.one_hot_encode_data import one_hot_encode_data


@pytest.fixture()
def mock_csv_file(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.csv file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    test_df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green'], 'Size': [10, 20, 30]})

    test_csv_path = temp_dir / 'mock.csv'
    test_df.to_csv(test_csv_path, index=False)

    return str(test_csv_path)


def test_one_hot_encode_data(mock_csv_file, tmp_path: pathlib.Path):
    """
    Checks whether categorical feature were transformed into one-hot representation.
    :param mock_csv_file: File path created in mock_csv_file().
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    output_path = tmp_path / "output.csv"

    # clean_data() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(one_hot_encode_data, [mock_csv_file, str(output_path)])

    assert result.exit_code == 0

    expected_data = {'Size': [10, 20, 30],
                     'Color_Blue': [False, True, False],
                     'Color_Green': [False, False, True],
                     'Color_Red': [True, False, False]}
    expected_df = pd.DataFrame(expected_data)

    test_df = pd.read_csv(output_path)

    assert test_df.equals(expected_df)
