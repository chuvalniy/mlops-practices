import pathlib

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data.encode_data import encode_data


@pytest.fixture()
def mock_csv_file(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.csv file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    test_df = pd.DataFrame(
        data=[['currently drinking', 'never smoked', 'abnormal', 'normal', 'Female'],
              ['not drinking', 'currently smoking', 'normal', 'abnormal', 'Male']],
        columns=['drinking', 'smoking', 'hear_left', 'hear_right', 'sex']
    )

    test_csv_path = temp_dir / 'mock.csv'
    test_df.to_csv(test_csv_path, index=False)

    return str(test_csv_path)


def test_encode_data(mock_csv_file, tmp_path: pathlib.Path):
    """
    Checks whether categorical feature were transformed into one-hot representation.
    :param mock_csv_file: File path created in mock_csv_file().
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    output_path = tmp_path / "output.csv"

    # encode_data() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(encode_data, [mock_csv_file, str(output_path)])

    assert result.exit_code == 0

    test_df = pd.read_csv(output_path)

    # Check if one-hot encoded columns are the same and have the same order with expected columns.
    expected_columns = ['drinking', 'smoking_currently smoking', 'smoking_never smoked', 'hear_left_abnormal',
                        'hear_left_normal', 'hear_right_abnormal', 'hear_right_normal', 'sex_Female', 'sex_Male']

    assert all(exp_col == test_col for exp_col, test_col in zip(expected_columns, test_df.columns))

    # Compare expected values with actual after one-hot encoding.
    expected_data = pd.DataFrame(
        data=[[0, False, True, True, False, False, True, True, False],
              [1, True, False, False, True, True, False, False, True]],
        columns=expected_columns
    )
    expected_df = pd.DataFrame(expected_data)

    print(expected_df)

    assert test_df.equals(expected_df)
